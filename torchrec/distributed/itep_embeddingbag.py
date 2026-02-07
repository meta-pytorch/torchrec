#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, OrderedDict, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import filter_state_dict
from torchrec.modules.itep_embedding_modules import (
    ITEPEmbeddingBagCollection,
    ITEPEmbeddingCollection,
)
from torchrec.modules.itep_modules import GenericITEPModule, RowwiseShardedITEPModule
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


@dataclass
class ITEPEmbeddingBagCollectionContext(EmbeddingBagCollectionContext):
    is_reindexed: bool = False


class ShardingTypeGroup(Enum):
    CW_GROUP = "column_wise_group"
    RW_GROUP = "row_wise_group"


SHARDING_TYPE_TO_GROUP: Dict[str, ShardingTypeGroup] = {
    ShardingType.ROW_WISE.value: ShardingTypeGroup.RW_GROUP,
    ShardingType.TABLE_ROW_WISE.value: ShardingTypeGroup.RW_GROUP,
    ShardingType.COLUMN_WISE.value: ShardingTypeGroup.CW_GROUP,
    ShardingType.TABLE_WISE.value: ShardingTypeGroup.CW_GROUP,
}


class ShardedITEPEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        KeyedTensor,
        ITEPEmbeddingBagCollectionContext,
    ]
):
    def __init__(
        self,
        module: ITEPEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env

        # Iteration counter for ITEP Module. Pinning on CPU because used for condition checking and checkpointing.
        self.register_buffer(
            "_iter", torch.tensor(0, dtype=torch.int64, device=torch.device("cpu"))
        )

        self._embedding_bag_collection: ShardedEmbeddingBagCollection = (
            ebc_sharder.shard(
                module._embedding_bag_collection,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
            )
        )

        self.table_name_to_sharding_type: Dict[str, str] = {}
        for table_name in table_name_to_parameter_sharding.keys():
            self.table_name_to_sharding_type[table_name] = (
                table_name_to_parameter_sharding[table_name].sharding_type
            )

        # Group lookups, table_name_to_unpruned_hash_sizes by sharding type and pass to separate itep modules
        (grouped_lookups, grouped_table_unpruned_size_map) = (
            self._group_lookups_and_table_unpruned_size_map(
                module._itep_module.table_name_to_unpruned_hash_sizes,
            )
        )

        # Instantiate ITEP Module in sharded case, re-using metadata from non-sharded case
        self._itep_module: GenericITEPModule = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=grouped_table_unpruned_size_map[
                ShardingTypeGroup.CW_GROUP
            ],
            lookups=grouped_lookups[ShardingTypeGroup.CW_GROUP],
            pruning_interval=module._itep_module.pruning_interval,
            enable_pruning=module._itep_module.enable_pruning,
            pg=env.process_group,
        )
        self._rowwise_itep_module: RowwiseShardedITEPModule = RowwiseShardedITEPModule(
            table_name_to_sharding_type=self.table_name_to_sharding_type,
            table_name_to_unpruned_hash_sizes=grouped_table_unpruned_size_map[
                ShardingTypeGroup.RW_GROUP
            ],
            lookups=grouped_lookups[ShardingTypeGroup.RW_GROUP],
            pruning_interval=module._itep_module.pruning_interval,
            enable_pruning=module._itep_module.enable_pruning,
            pg=env.process_group,
        )

    def prefetch(
        self,
        dist_input: KJTList,
        forward_stream: Optional[Union[torch.cuda.Stream, torch.mtia.Stream]] = None,
        ctx: Optional[ITEPEmbeddingBagCollectionContext] = None,
    ) -> None:
        assert (
            ctx is not None
        ), "ITEP Prefetch call requires ITEPEmbeddingBagCollectionContext"
        dist_input = self._reindex(dist_input)
        ctx.is_reindexed = True
        self._embedding_bag_collection.prefetch(dist_input, forward_stream, ctx)

    def input_dist(
        self,
        ctx: ITEPEmbeddingBagCollectionContext,
        features: KeyedJaggedTensor,
        force_insert: bool = False,
    ) -> Awaitable[Awaitable[KJTList]]:
        return self._embedding_bag_collection.input_dist(ctx, features)

    def _reindex(self, dist_input: KJTList) -> KJTList:
        for i, (sharding, features) in enumerate(
            zip(
                self._embedding_bag_collection._sharding_types,
                dist_input,
            )
        ):
            if SHARDING_TYPE_TO_GROUP[sharding] == ShardingTypeGroup.CW_GROUP:
                remapped_kjt = self._itep_module(features, self._iter.item())
            else:
                remapped_kjt = self._rowwise_itep_module(features, self._iter.item())
            dist_input[i] = remapped_kjt
        return dist_input

    def compute(
        self,
        ctx: ITEPEmbeddingBagCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        # We need to explicitly move iter to CPU since it might be moved to GPU
        # after __init__. This should be done once.
        self._iter = self._iter.cpu()

        if not ctx.is_reindexed:
            dist_input = self._reindex(dist_input)
            ctx.is_reindexed = True

        self._iter += 1
        return self._embedding_bag_collection.compute(ctx, dist_input)

    def output_dist(
        self,
        ctx: ITEPEmbeddingBagCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:

        ebc_awaitable = self._embedding_bag_collection.output_dist(ctx, output)
        return ebc_awaitable

    def compute_and_output_dist(
        self, ctx: ITEPEmbeddingBagCollectionContext, input: KJTList
    ) -> LazyAwaitable[KeyedTensor]:
        # We need to explicitly move iter to CPU since it might be moved to GPU
        # after __init__. This should be done once.
        self._iter = self._iter.cpu()

        # Insert forward() function of GenericITEPModule into compute_and_output_dist()
        for i, (sharding, features) in enumerate(
            zip(
                self._embedding_bag_collection._sharding_types,
                input,
            )
        ):
            if SHARDING_TYPE_TO_GROUP[sharding] == ShardingTypeGroup.CW_GROUP:
                remapped_kjt = self._itep_module(features, self._iter.item())
            else:
                remapped_kjt = self._rowwise_itep_module(features, self._iter.item())
            input[i] = remapped_kjt
        self._iter += 1
        ebc_awaitable = self._embedding_bag_collection.compute_and_output_dist(
            ctx, input
        )
        return ebc_awaitable

    def create_context(self) -> ITEPEmbeddingBagCollectionContext:
        return ITEPEmbeddingBagCollectionContext()

    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        self._iter = state_dict["_iter"]
        for name, child_module in self._modules.items():
            if child_module is not None:
                missing, unexpected = child_module.load_state_dict(
                    filter_state_dict(state_dict, name),
                    strict,
                )
                missing_keys.extend(missing)
                unexpected_keys.extend(unexpected)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def _group_lookups_and_table_unpruned_size_map(
        self, table_name_to_unpruned_hash_sizes: Dict[str, int]
    ) -> Tuple[
        Dict[ShardingTypeGroup, List[nn.Module]],
        Dict[ShardingTypeGroup, Dict[str, int]],
    ]:
        """
        Group ebc lookups and table_name_to_unpruned_hash_sizes by sharding types.
        CW and TW are grouped into CW_GROUP, RW and TWRW are grouped into RW_GROUP.

        Return a tuple of (grouped_lookups, grouped _table_unpruned_size_map)
        """
        grouped_lookups: Dict[ShardingTypeGroup, List[nn.Module]] = defaultdict(list)
        grouped_table_unpruned_size_map: Dict[ShardingTypeGroup, Dict[str, int]] = (
            defaultdict(dict)
        )
        for sharding_type, lookup in zip(
            self._embedding_bag_collection._sharding_types,
            self._embedding_bag_collection._lookups,
        ):
            sharding_group = SHARDING_TYPE_TO_GROUP[sharding_type]
            # group lookups
            grouped_lookups[sharding_group].append(lookup)
            # group table_name_to_unpruned_hash_sizes
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            for emb_config in lookup.grouped_configs:
                for table in emb_config.embedding_tables:
                    if table.name in table_name_to_unpruned_hash_sizes.keys():
                        grouped_table_unpruned_size_map[sharding_group][table.name] = (
                            table_name_to_unpruned_hash_sizes[table.name]
                        )

        return grouped_lookups, grouped_table_unpruned_size_map


class ITEPEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[ITEPEmbeddingBagCollection]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingBagCollectionSharder = (
            ebc_sharder
            or EmbeddingBagCollectionSharder(
                qcomm_codecs_registry=self.qcomm_codecs_registry
            )
        )

    def shard(
        self,
        module: ITEPEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedITEPEmbeddingBagCollection:

        # Enforce GPU for ITEPEmbeddingBagCollection
        if device is None:
            device = torch.device("cuda")

        return ShardedITEPEmbeddingBagCollection(
            module,
            params,
            ebc_sharder=self._ebc_sharder,
            env=env,
            device=device,
        )

    def shardable_parameters(
        self, module: ITEPEmbeddingBagCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ebc_sharder.shardable_parameters(module._embedding_bag_collection)

    @property
    def module_type(self) -> Type[ITEPEmbeddingBagCollection]:
        return ITEPEmbeddingBagCollection

    @property
    def sharded_module_type(self) -> Type[ShardedITEPEmbeddingBagCollection]:
        return ShardedITEPEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = list(SHARDING_TYPE_TO_GROUP.keys())
        return types


class ITEPEmbeddingCollectionContext(EmbeddingCollectionContext):

    def __init__(self) -> None:
        super().__init__()
        self.is_reindexed: bool = False
        self.table_name_to_unpruned_hash_sizes: Dict[str, int] = {}


class ShardedITEPEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        Dict[str, JaggedTensor],
        ITEPEmbeddingCollectionContext,
    ]
):
    def __init__(
        self,
        module: ITEPEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingCollectionSharder,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env
        self.table_name_to_unpruned_hash_sizes: Dict[str, int] = (
            module._itep_module.table_name_to_unpruned_hash_sizes
        )

        # Iteration counter for ITEP Module. Pinning on CPU because used for condition checking and checkpointing.
        self.register_buffer(
            "_iter", torch.tensor(0, dtype=torch.int64, device=torch.device("cpu"))
        )

        self._embedding_collection: ShardedEmbeddingCollection = ebc_sharder.shard(
            module._embedding_collection,
            table_name_to_parameter_sharding,
            env=env,
            device=device,
        )

        self.table_name_to_sharding_type: Dict[str, str] = {}
        for table_name in table_name_to_parameter_sharding.keys():
            self.table_name_to_sharding_type[table_name] = (
                table_name_to_parameter_sharding[table_name].sharding_type
            )

        # Group lookups, table_name_to_unpruned_hash_sizes by sharding type and pass to separate itep modules
        (grouped_lookups, grouped_table_unpruned_size_map) = (
            self._group_lookups_and_table_unpruned_size_map(
                module._itep_module.table_name_to_unpruned_hash_sizes,
            )
        )

        # Instantiate ITEP Module in sharded case, re-using metadata from non-sharded case
        self._itep_module: GenericITEPModule = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=grouped_table_unpruned_size_map[
                ShardingTypeGroup.CW_GROUP
            ],
            lookups=grouped_lookups[ShardingTypeGroup.CW_GROUP],
            pruning_interval=module._itep_module.pruning_interval,
            enable_pruning=module._itep_module.enable_pruning,
            pg=env.process_group,
        )
        self._rowwise_itep_module: RowwiseShardedITEPModule = RowwiseShardedITEPModule(
            table_name_to_unpruned_hash_sizes=grouped_table_unpruned_size_map[
                ShardingTypeGroup.RW_GROUP
            ],
            lookups=grouped_lookups[ShardingTypeGroup.RW_GROUP],
            pruning_interval=module._itep_module.pruning_interval,
            table_name_to_sharding_type=self.table_name_to_sharding_type,
            enable_pruning=module._itep_module.enable_pruning,
            pg=env.process_group,
        )

    def input_dist(
        self,
        ctx: ITEPEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
        force_insert: bool = False,
    ) -> Awaitable[Awaitable[KJTList]]:

        ctx.table_name_to_unpruned_hash_sizes = self.table_name_to_unpruned_hash_sizes
        return self._embedding_collection.input_dist(ctx, features)

    def compute(
        self,
        ctx: ITEPEmbeddingCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        # We need to explicitly move iter to CPU since it might be moved to GPU
        # after __init__. This should be done once.
        self._iter = self._iter.cpu()

        for i, (sharding, features) in enumerate(
            zip(
                self._embedding_collection._sharding_type_to_sharding.keys(),
                dist_input,
            )
        ):
            if SHARDING_TYPE_TO_GROUP[sharding] == ShardingTypeGroup.CW_GROUP:
                remapped_kjt = self._itep_module(features, self._iter.item())
            else:
                remapped_kjt = self._rowwise_itep_module(features, self._iter.item())
            dist_input[i] = remapped_kjt
        self._iter += 1
        return self._embedding_collection.compute(ctx, dist_input)

    def output_dist(
        self,
        ctx: ITEPEmbeddingCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:

        ec_awaitable = self._embedding_collection.output_dist(ctx, output)
        return ec_awaitable

    def compute_and_output_dist(
        self, ctx: ITEPEmbeddingCollectionContext, input: KJTList
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        # We need to explicitly move iter to CPU since it might be moved to GPU
        # after __init__. This should be done once.
        self._iter = self._iter.cpu()

        # Insert forward() function of GenericITEPModule into compute_and_output_dist()
        """ """
        for i, (sharding, features) in enumerate(
            zip(
                self._embedding_collection._sharding_type_to_sharding.keys(),
                input,
            )
        ):
            if SHARDING_TYPE_TO_GROUP[sharding] == ShardingTypeGroup.CW_GROUP:
                remapped_kjt = self._itep_module(features, self._iter.item())
            else:
                remapped_kjt = self._rowwise_itep_module(features, self._iter.item())
            input[i] = remapped_kjt
        self._iter += 1
        ec_awaitable = self._embedding_collection.compute_and_output_dist(ctx, input)
        return ec_awaitable

    def create_context(self) -> ITEPEmbeddingCollectionContext:
        return ITEPEmbeddingCollectionContext()

    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        self._iter = state_dict["_iter"]
        for name, child_module in self._modules.items():
            if child_module is not None:
                missing, unexpected = child_module.load_state_dict(
                    filter_state_dict(state_dict, name),
                    strict,
                )
                missing_keys.extend(missing)
                unexpected_keys.extend(unexpected)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def _group_lookups_and_table_unpruned_size_map(
        self, table_name_to_unpruned_hash_sizes: Dict[str, int]
    ) -> Tuple[
        Dict[ShardingTypeGroup, List[nn.Module]],
        Dict[ShardingTypeGroup, Dict[str, int]],
    ]:
        """
        Group ebc lookups and table_name_to_unpruned_hash_sizes by sharding types.
        CW and TW are grouped into CW_GROUP, RW and TWRW are grouped into RW_GROUP.

        Return a tuple of (grouped_lookups, grouped _table_unpruned_size_map)
        """
        grouped_lookups: Dict[ShardingTypeGroup, List[nn.Module]] = defaultdict(list)
        grouped_table_unpruned_size_map: Dict[ShardingTypeGroup, Dict[str, int]] = (
            defaultdict(dict)
        )
        for sharding_type, lookup in zip(
            self._embedding_collection._sharding_types,
            self._embedding_collection._lookups,
        ):
            sharding_group = SHARDING_TYPE_TO_GROUP[sharding_type]
            # group lookups
            grouped_lookups[sharding_group].append(lookup)
            # group table_name_to_unpruned_hash_sizes
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            for emb_config in lookup.grouped_configs:
                for table in emb_config.embedding_tables:
                    if table.name in table_name_to_unpruned_hash_sizes.keys():
                        grouped_table_unpruned_size_map[sharding_group][table.name] = (
                            table_name_to_unpruned_hash_sizes[table.name]
                        )

        return grouped_lookups, grouped_table_unpruned_size_map


class ITEPEmbeddingCollectionSharder(BaseEmbeddingSharder[ITEPEmbeddingCollection]):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingCollectionSharder = (
            ebc_sharder
            or EmbeddingCollectionSharder(
                qcomm_codecs_registry=self.qcomm_codecs_registry
            )
        )

    def shard(
        self,
        module: ITEPEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedITEPEmbeddingCollection:

        # Enforce GPU for ITEPEmbeddingBagCollection
        if device is None:
            device = torch.device("cuda")

        return ShardedITEPEmbeddingCollection(
            module,
            params,
            ebc_sharder=self._ebc_sharder,
            env=env,
            device=device,
        )

    def shardable_parameters(
        self, module: ITEPEmbeddingCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ebc_sharder.shardable_parameters(module._embedding_collection)

    @property
    def module_type(self) -> Type[ITEPEmbeddingCollection]:
        return ITEPEmbeddingCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = list(SHARDING_TYPE_TO_GROUP.keys())
        return types
