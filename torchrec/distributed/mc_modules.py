#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import itertools
import logging
import math
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._shard.sharded_tensor import Shard, ShardMetadata
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    GroupedEmbeddingConfig,
    KJTList,
    ListOfKJTList,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    RwSequenceEmbeddingDist,
    RwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sharding import (
    BaseRwEmbeddingSharding,
    InferRwSparseFeaturesDist,
    RwSparseFeaturesDist,
)
from torchrec.distributed.sharding.sequence_sharding import (
    InferSequenceShardingContext,
    SequenceShardingContext,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    NullShardedModuleContext,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.modules.utils import construct_jagged_tensors
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.streamable import Multistreamable


@dataclass
class EmbeddingCollectionContext(Multistreamable):
    sharding_contexts: List[
        Union[InferSequenceShardingContext, SequenceShardingContext]
    ]
    input_features: List[KeyedJaggedTensor] = field(default_factory=list)
    # VBE-Attributes for EBC
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None
    variable_batch_per_feature: bool = False

    def record_stream(self, stream: torch.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)
        for f in self.input_features:
            # pyrefly: ignore[bad-argument-type]
            f.record_stream(stream)
        if self.inverse_indices is not None:
            self.inverse_indices[1].record_stream(stream)


class ManagedCollisionCollectionContext(EmbeddingCollectionContext):
    pass


@torch.fx.wrap
def _fx_global_to_local_index(
    feature_dict: Dict[str, JaggedTensor], feature_to_offset: Dict[str, int]
) -> Dict[str, JaggedTensor]:
    for feature, jt in feature_dict.items():
        jt._values = jt.values() - feature_to_offset[feature]
    return feature_dict


@torch.fx.wrap
def _fx_jt_dict_add_offset(
    feature_dict: Dict[str, JaggedTensor], feature_to_offset: Dict[str, int]
) -> Dict[str, JaggedTensor]:
    for feature, jt in feature_dict.items():
        jt._values = jt.values() + feature_to_offset[feature]
    return feature_dict


@torch.fx.wrap
def _get_length_per_key(kjt: KeyedJaggedTensor) -> torch.Tensor:
    return torch.tensor(kjt.length_per_key())


logger: logging.Logger = logging.getLogger(__name__)


class ManagedCollisionCollectionAwaitable(LazyAwaitable[KeyedJaggedTensor]):
    def __init__(
        self,
        awaitables_per_sharding: List[Awaitable[torch.Tensor]],
        features_per_sharding: List[KeyedJaggedTensor],
        embedding_names_per_sharding: List[List[str]],
        need_indices: bool = False,
        features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self._awaitables_per_sharding = awaitables_per_sharding
        self._features_per_sharding = features_per_sharding
        self._need_indices = need_indices
        self._features_to_permute_indices = features_to_permute_indices
        self._embedding_names_per_sharding = embedding_names_per_sharding
        self._reverse_indices = reverse_indices

    def _wait_impl(self) -> KeyedJaggedTensor:
        jt_dict: Dict[str, JaggedTensor] = {}
        for i, (w, f, e) in enumerate(
            zip(
                self._awaitables_per_sharding,
                self._features_per_sharding,
                self._embedding_names_per_sharding,
            )
        ):
            reverse_indices = (
                self._reverse_indices[i] if self._reverse_indices else None
            )

            jt_dict.update(
                construct_jagged_tensors(
                    embeddings=w.wait(),
                    features=f,
                    embedding_names=e,
                    need_indices=self._need_indices,
                    features_to_permute_indices=self._features_to_permute_indices,
                    reverse_indices=reverse_indices,
                )
            )
            # TODO: find better solution
            for jt in jt_dict.values():
                jt._values = jt.values().flatten()
        return KeyedJaggedTensor.from_jt_dict(jt_dict)


def create_mc_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[
    SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
]:
    if sharding_type == ShardingType.ROW_WISE.value:
        return RwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
        )
    else:
        raise ValueError(f"Sharding not supported {sharding_type}")


class ShardedManagedCollisionCollection(
    ShardedModule[
        KJTList,
        KJTList,
        KeyedJaggedTensor,
        ManagedCollisionCollectionContext,
    ]
):
    def __init__(
        self,
        module: ManagedCollisionCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: torch.device,
        embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        # pyrefly: ignore[missing-attribute]
        super().__init__()
        self.need_preprocess: bool = module.need_preprocess
        self._device = device
        self._env = env
        self._table_name_to_parameter_sharding: Dict[str, ParameterSharding] = (
            copy.deepcopy(table_name_to_parameter_sharding)
        )
        # TODO: create a MCSharding type instead of leveraging EmbeddingSharding
        self._embedding_shardings = embedding_shardings

        self._embedding_names_per_sharding: List[List[str]] = []
        for sharding in self._embedding_shardings:
            # TODO: support TWRW sharding
            assert isinstance(
                sharding, BaseRwEmbeddingSharding
            ), "Only ROW_WISE sharding is supported."
            self._embedding_names_per_sharding.append(sharding.embedding_names())

        self._feature_to_table: Dict[str, str] = module._feature_to_table
        self._table_to_features: Dict[str, List[str]] = module._table_to_features
        self._table_name_to_config: Dict[str, BaseEmbeddingConfig] = (
            module._table_name_to_config
        )
        self._has_uninitialized_input_dists: bool = True
        self._input_dists: List[nn.Module] = []
        self._managed_collision_modules = nn.ModuleDict()
        self._create_managed_collision_modules(module)
        self._output_dists: List[nn.Module] = []
        self._create_output_dists()
        self._use_index_dedup = use_index_dedup
        self._initialize_torch_state()
        self.post_lookup_tracker_fn: Optional[
            Callable[
                [
                    KeyedJaggedTensor,
                    torch.Tensor,
                    Optional[nn.Module],
                    Optional[torch.Tensor],
                    Optional[torch.Tensor],
                ],
                None,
            ]
        ] = None

    def _initialize_torch_state(self) -> None:
        self._model_parallel_mc_buffer_name_to_sharded_tensor = OrderedDict()
        shardable_params = set(
            self.sharded_parameter_names(prefix="_managed_collision_modules")
        )

        for fqn, tensor in self.state_dict().items():
            if fqn not in shardable_params:
                continue
            table_name = fqn.split(".")[
                1
            ]  #  "_managed_collision_modules.<table_name>.<param_name>"
            shard_offset, shard_size, global_size = self._mc_module_name_shard_metadata[
                table_name
            ]
            sharded_sizes = list(tensor.shape)
            sharded_sizes[0] = shard_size
            shard_offsets = [0] * len(sharded_sizes)
            shard_offsets[0] = shard_offset
            global_sizes = list(tensor.shape)
            global_sizes[0] = global_size

            self._model_parallel_mc_buffer_name_to_sharded_tensor[fqn] = (
                ShardedTensor._init_from_local_shards(
                    [
                        Shard(
                            tensor=tensor,
                            metadata=ShardMetadata(
                                shard_offsets=shard_offsets,
                                shard_sizes=sharded_sizes,
                                placement=(f"rank:{self._env.rank}/{tensor.device}"),
                            ),
                        )
                    ],
                    torch.Size(global_sizes),
                    process_group=self._env.process_group,
                )
            )

        def _post_state_dict_hook(
            module: ShardedManagedCollisionCollection,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            for (
                mc_buffer_name,
                sharded_tensor,
            ) in module._model_parallel_mc_buffer_name_to_sharded_tensor.items():
                destination_key = f"{prefix}{mc_buffer_name}"
                destination[destination_key] = sharded_tensor

        def _load_state_dict_pre_hook(
            module: "ShardedManagedCollisionCollection",
            state_dict: Dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> None:
            for (
                mc_buffer_name,
                _sharded_tensor,
            ) in module._model_parallel_mc_buffer_name_to_sharded_tensor.items():
                key = f"{prefix}{mc_buffer_name}"
                if key in state_dict:
                    if isinstance(state_dict[key], ShardedTensor):
                        local_shards = state_dict[key].local_shards()
                        state_dict[key] = local_shards[0].tensor
                    else:
                        raise RuntimeError(
                            f"Unexpected state_dict key type {type(state_dict[key])} found for {key}"
                        )

        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            _load_state_dict_pre_hook, with_module=True
        )

    def _create_managed_collision_modules(
        self, module: ManagedCollisionCollection
    ) -> None:

        self._mc_module_name_shard_metadata: DefaultDict[str, List[int]] = defaultdict()
        # To map mch output indices from local to global. key: table_name
        self._table_to_offset: Dict[str, int] = {}

        # the split sizes of tables belonging to each sharding. outer len is # shardings
        self._sharding_per_table_feature_splits: List[List[int]] = []
        self._input_size_per_table_feature_splits: List[List[int]] = []
        # the split sizes of features per sharding. len is # shardings
        self._sharding_feature_splits: List[int] = []
        # the split sizes of features per table. len is # tables sum over all shardings
        self._table_feature_splits: List[int] = []
        self._feature_names: List[str] = []

        # table names of each sharding
        self._sharding_tables: List[List[str]] = []
        self._sharding_features: List[List[str]] = []

        for sharding in self._embedding_shardings:
            assert isinstance(sharding, BaseRwEmbeddingSharding)
            self._sharding_tables.append([])
            self._sharding_features.append([])
            self._sharding_per_table_feature_splits.append([])
            self._input_size_per_table_feature_splits.append([])

            grouped_embedding_configs: List[GroupedEmbeddingConfig] = (
                sharding._grouped_embedding_configs
            )
            self._sharding_feature_splits.append(len(sharding.feature_names()))

            num_sharding_features = 0
            for group_config in grouped_embedding_configs:
                for table in group_config.embedding_tables:
                    # pyrefly: ignore[missing-attribute]
                    new_min_output_id = table.local_metadata.shard_offsets[0]
                    # pyrefly: ignore[missing-attribute]
                    new_range_size = table.local_metadata.shard_sizes[0]
                    output_segments = [
                        x.shard_offsets[0]
                        # pyrefly: ignore[missing-attribute]
                        for x in table.global_metadata.shards_metadata
                    ] + [table.num_embeddings]
                    mc_module = module._managed_collision_modules[table.name]

                    self._sharding_tables[-1].append(table.name)
                    self._sharding_features[-1].extend(table.feature_names)
                    self._feature_names.extend(table.feature_names)
                    self._managed_collision_modules[table.name] = (
                        # pyrefly: ignore[not-callable]
                        mc_module.rebuild_with_output_id_range(
                            output_id_range=(
                                new_min_output_id,
                                new_min_output_id + new_range_size,
                            ),
                            output_segments=output_segments,
                            device=self._device,
                        )
                    )
                    # pyrefly: ignore[not-callable]
                    zch_size = self._managed_collision_modules[table.name].output_size()
                    # pyrefly: ignore[not-callable]
                    input_size = self._managed_collision_modules[
                        table.name
                    ].input_size()
                    zch_size_by_rank = [
                        torch.zeros(1, dtype=torch.int64, device=self._device)
                        for _ in range(self._env.world_size)
                    ]
                    if self.training and self._env.world_size > 1:
                        dist.all_gather(
                            zch_size_by_rank,
                            torch.tensor(
                                [zch_size], dtype=torch.int64, device=self._device
                            ),
                            group=self._env.process_group,
                        )
                    else:
                        zch_size_by_rank[0] = torch.tensor(
                            [zch_size], dtype=torch.int64, device=self._device
                        )

                    # Calculate the sum of all ZCH sizes from rank 0 to list
                    # index. The last item is the sum of all elements in zch_size_by_rank
                    zch_size_cumsum = torch.cumsum(
                        torch.cat(zch_size_by_rank), dim=0
                    ).tolist()

                    zch_size_sum_before_this_rank = (
                        zch_size_cumsum[self._env.rank] - zch_size
                    )
                    # pyrefly: ignore[unsupported-operation]
                    self._mc_module_name_shard_metadata[table.name] = (
                        zch_size_sum_before_this_rank,
                        zch_size,
                        zch_size_cumsum[-1],
                    )
                    self._table_to_offset[table.name] = new_min_output_id

                    self._table_feature_splits.append(len(table.feature_names))
                    self._sharding_per_table_feature_splits[-1].append(
                        self._table_feature_splits[-1]
                    )
                    self._input_size_per_table_feature_splits[-1].append(
                        input_size,
                    )
                    num_sharding_features += self._table_feature_splits[-1]

            assert num_sharding_features == len(
                sharding.feature_names()
            ), f"Shared feature is not supported. {num_sharding_features=}, {self._sharding_per_table_feature_splits[-1]=}"

            if self._sharding_features[-1] != sharding.feature_names():
                logger.warning(
                    "The order of tables of this sharding is altered due to grouping: "
                    f"{self._sharding_features[-1]=} vs {sharding.feature_names()=}"
                )

        logger.info(f"{self._table_feature_splits=}")
        logger.info(f"{self._sharding_per_table_feature_splits=}")
        logger.info(f"{self._input_size_per_table_feature_splits=}")
        logger.info(f"{self._feature_names=}")
        logger.info(f"{self._table_to_offset=}")
        logger.info(f"{self._sharding_tables=}")
        logger.info(f"{self._sharding_features=}")

    def _create_input_dists(
        self,
        input_feature_names: List[str],
    ) -> None:
        for sharding, sharding_features in zip(
            self._embedding_shardings,
            self._sharding_features,
        ):
            assert isinstance(sharding, BaseRwEmbeddingSharding)
            feature_num_buckets: List[int] = [
                # pyrefly: ignore[not-callable]
                self._managed_collision_modules[self._feature_to_table[f]].buckets()
                for f in sharding_features
            ]

            input_sizes: List[int] = [
                # pyrefly: ignore[not-callable]
                self._managed_collision_modules[self._feature_to_table[f]].input_size()
                for f in sharding_features
            ]

            feature_hash_sizes: List[int] = []
            feature_total_num_buckets: List[int] = []
            for input_size, num_buckets in zip(
                input_sizes,
                feature_num_buckets,
            ):
                feature_hash_sizes.append(input_size)
                feature_total_num_buckets.append(num_buckets)

            input_dist = RwSparseFeaturesDist(
                # pyrefly: ignore[bad-argument-type]
                pg=sharding._pg,
                num_features=sharding._get_num_features(),
                feature_hash_sizes=feature_hash_sizes,
                feature_total_num_buckets=feature_total_num_buckets,
                device=sharding._device,
                is_sequence=True,
                has_feature_processor=sharding._has_feature_processor,
                need_pos=False,
                keep_original_indices=True,
            )
            self._input_dists.append(input_dist)

        #  `_features_order`.
        self._features_order: List[int] = []
        for f in self._feature_names:
            #  `append`.
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(input_feature_names)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=self._device, dtype=torch.int32),
            persistent=False,
        )
        if self._use_index_dedup:
            self._create_dedup_indices()

    def _create_output_dists(
        self,
    ) -> None:
        for sharding in self._embedding_shardings:
            assert isinstance(sharding, BaseRwEmbeddingSharding)
            self._output_dists.append(
                RwSequenceEmbeddingDist(
                    # pyrefly: ignore[bad-argument-type]
                    sharding._pg,
                    sharding._get_num_features(),
                    sharding._device,
                )
            )

    def _create_dedup_indices(self) -> None:
        # validate we can linearize the features irrespective of feature split
        assert (
            list(
                itertools.accumulate(
                    [
                        hash_input
                        for input_split in self._input_size_per_table_feature_splits
                        for hash_input in input_split
                    ]
                )
            )[-1]
            <= torch.iinfo(torch.int64).max
        ), "EC Dedup requires the mc collection to have a cumuluative 'hash_input_size' kwarg to be less than max int64.  Please reduce values of individual tables to meet this constraint (ie. 2**54 is typically a good value)."
        for i, (feature_splits, input_splits) in enumerate(
            zip(
                self._sharding_per_table_feature_splits,
                self._input_size_per_table_feature_splits,
            )
        ):
            cum_f = 0
            cum_i = 0
            hash_offsets = []
            feature_offsets = []
            N = math.ceil(math.log2(len(feature_splits)))
            for features, hash_size in zip(feature_splits, input_splits):
                hash_offsets += [cum_i for _ in range(features)]
                feature_offsets += [cum_f for _ in range(features)]
                cum_f += features
                cum_i += (2 ** (63 - N) - 1) if hash_size == 0 else hash_size
                assert (
                    cum_i <= torch.iinfo(torch.int64).max
                ), f"Index exceeds max int64, {cum_i=}"
            hash_offsets += [cum_i]
            feature_offsets += [cum_f]
            self.register_buffer(
                "_dedup_hash_offsets_{}".format(i),
                torch.tensor(hash_offsets, dtype=torch.int64, device=self._device),
                persistent=False,
            )
            self.register_buffer(
                "_dedup_feature_offsets_{}".format(i),
                torch.tensor(feature_offsets, dtype=torch.int64, device=self._device),
                persistent=False,
            )

    def _dedup_indices(
        self,
        ctx: ManagedCollisionCollectionContext,
        features: List[KeyedJaggedTensor],
    ) -> List[KeyedJaggedTensor]:
        features_by_sharding = []

        for i, kjt in enumerate(features):
            hash_offsets = self.get_buffer(f"_dedup_hash_offsets_{i}")
            feature_offsets = self.get_buffer(f"_dedup_feature_offsets_{i}")
            (
                lengths,
                offsets,
                unique_indices,
                reverse_indices,
            ) = torch.ops.fbgemm.jagged_unique_indices(
                hash_offsets,
                feature_offsets,
                kjt.offsets().to(torch.int64),
                kjt.values().to(torch.int64),
            )
            dedup_features = KeyedJaggedTensor(
                keys=kjt.keys(),
                lengths=lengths,
                offsets=offsets,
                values=unique_indices,
            )

            ctx.input_features.append(kjt)
            # pyrefly: ignore[missing-attribute]
            ctx.reverse_indices.append(reverse_indices)
            features_by_sharding.append(dedup_features)
        return features_by_sharding

    # pyrefly: ignore[bad-override]
    def input_dist(
        self,
        ctx: ManagedCollisionCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        if self._has_uninitialized_input_dists:
            self._create_input_dists(input_feature_names=features.keys())
            self._has_uninitialized_input_dists = False

        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    #  `Union[Module, Tensor]`.
                    self._features_order,
                    #  but got `Union[Module, Tensor]`.
                    # pyrefly: ignore[bad-argument-type]
                    self._features_order_tensor,
                )

            feature_splits: List[KeyedJaggedTensor] = []
            if self.need_preprocess:
                # NOTE: No shared features allowed!
                assert (
                    len(self._sharding_feature_splits) == 1
                ), "Preprocing only support single sharding type (row-wise)"
                table_splits = features.split(self._table_feature_splits)
                ti: int = 0
                for i, tables in enumerate(self._sharding_tables):
                    output: Dict[str, JaggedTensor] = {}
                    stride_per_key_per_rank_list: List[List[int]] = []
                    for table in tables:
                        kjt: KeyedJaggedTensor = table_splits[ti]
                        mc_module = self._managed_collision_modules[table]
                        # TODO: change to Dict[str, Tensor]
                        mc_input: Dict[str, JaggedTensor] = {
                            table: JaggedTensor(
                                values=kjt.values(),
                                lengths=kjt.lengths(),
                            )
                        }
                        # pyrefly: ignore[not-callable]
                        mc_input = mc_module.preprocess(mc_input)
                        output.update(mc_input)
                        stride_per_key_per_rank_list += kjt.stride_per_key_per_rank()
                        ti += 1
                    shard_kjt = KeyedJaggedTensor(
                        keys=self._sharding_features[i],
                        values=torch.cat([jt.values() for jt in output.values()]),
                        lengths=torch.cat([jt.lengths() for jt in output.values()]),
                        stride_per_key_per_rank=stride_per_key_per_rank_list or None,
                    )
                    feature_splits.append(shard_kjt)
            else:
                feature_splits = features.split(self._sharding_feature_splits)

            if self._use_index_dedup:
                feature_splits = self._dedup_indices(ctx, feature_splits)

            awaitables = []
            for feature_split, input_dist in zip(feature_splits, self._input_dists):
                awaitables.append(input_dist(feature_split))
                ctx.sharding_contexts.append(
                    SequenceShardingContext(
                        unbucketize_permute_tensor=(
                            input_dist.unbucketize_permute_tensor
                            if isinstance(input_dist, RwSparseFeaturesDist)
                            else None
                        ),
                        # For VBE-Support for EBC
                        # Split of the feature before all2all
                        batch_size_per_feature_pre_a2a=feature_split.stride_per_key(),
                        variable_batch_per_feature=feature_split.variable_stride_per_key(),
                        # For VBE-Support for EC
                        features_before_input_dist=features,
                    )
                )

        return KJTListSplitsAwaitable(awaitables, ctx)

    def _kjt_list_to_tensor_list(
        self,
        kjt_list: KJTList,
    ) -> List[torch.Tensor]:
        remapped_ids_ret: List[torch.Tensor] = []
        # TODO: find a better solution, could be padding
        for kjt, tables, splits in zip(
            kjt_list, self._sharding_tables, self._sharding_per_table_feature_splits
        ):
            if len(splits) > 1:
                feature_splits = kjt.split(splits)
                vals: List[torch.Tensor] = []
                # assert len(feature_splits) == len(sharding.embedding_tables())
                for feature_split, table in zip(feature_splits, tables):
                    offset = self._table_to_offset[table]
                    vals.append(feature_split.values() + offset)
                remapped_ids_ret.append(torch.cat(vals).view(-1, 1))
            else:
                remapped_ids_ret.append(
                    (kjt.values() + self._table_to_offset[tables[0]]).unsqueeze(-1)
                )
        return remapped_ids_ret

    def global_to_local_index(
        self,
        jt_dict: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        for table, jt in jt_dict.items():
            jt._values = jt.values() - self._table_to_offset[table]
        return jt_dict

    def _retrieve_and_track_hash_zch_identities_and_metadata(
        self,
        mcm: nn.Module,
        mc_input: Dict[str, JaggedTensor],
        indices: torch.Tensor,
    ) -> None:
        if self.post_lookup_tracker_fn is None:
            return
        if not hasattr(mcm, "_hash_zch_identities"):
            return
        # _hash_zch_identities should always exist but _hash_zch_runtime_meta is optional
        runtime_meta = None
        if (
            hasattr(mcm, "_hash_zch_runtime_meta")
            and mcm._hash_zch_runtime_meta is not None
        ):
            # pyrefly: ignore[not-callable]
            runtime_meta = mcm._hash_zch_runtime_meta.index_select(dim=0, index=indices)
        self.post_lookup_tracker_fn(
            KeyedJaggedTensor.from_jt_dict(mc_input),
            torch.empty(0),
            None,
            # pyrefly: ignore[not-callable]
            mcm._hash_zch_identities.index_select(dim=0, index=indices),
            runtime_meta,
        )

    def get_lookup_value(
        self, table: str, features: KeyedJaggedTensor
    ) -> Dict[str, JaggedTensor]:

        mcm = self._managed_collision_modules[table]
        # only turn on include_readonly_suffix_feature when at least one feature in the table ends with the readonly suffix
        include_readonly_suffix_feature = any(
            1
            for feature_name in features.keys()
            # pyrefly: ignore[bad-argument-type]
            if feature_name.lower().endswith(mcm.readable_suffix)
        )

        # When we turn on include_readonly_suffix_feature, those features will not contribute to the ZCH frequency counter, only non-readonly features will be consider for insert, eviction and stats logging
        # pyrefly: ignore[not-callable]
        if mcm._enable_per_feature_lookups and include_readonly_suffix_feature:
            # In the original lookup call, it will not distinct feature values and do remapper at one call for all features. In that way we can not define read-only feature, so here we pass in per key KJT to provide more control
            mc_input = features.to_dict()
            # pyrefly: ignore[not-callable]
            mc_input = mcm.profile(mc_input)
            # pyrefly: ignore[not-callable]
            mc_input = mcm.remap(mc_input)
            # This is for the purpose of remap the global index back to local index since the offset is key by table
            mc_input_unify = {
                table: JaggedTensor(
                    values=torch.cat([jt.values() for jt in mc_input.values()]),
                    lengths=torch.cat([jt.lengths() for jt in mc_input.values()]),
                )
            }
            mc_input = self.global_to_local_index(mc_input_unify)
        else:
            # this is the default behavior, we will do remapper for all features at one call
            mc_input: Dict[str, JaggedTensor] = {
                table: JaggedTensor(
                    values=features.values(),
                    lengths=features.lengths(),
                    # TODO: improve this temp solution by passing real weights, this is for eviction purpose since after we unify all feature to one key, we lost the original feature boundary information for per feature eviction
                    weights=torch.tensor(features.length_per_key()),
                )
            }
            # pyrefly: ignore[not-callable]
            mc_input = mcm.profile(mc_input)
            # pyrefly: ignore[not-callable]
            mc_input = mcm.remap(mc_input)
            mc_input = self.global_to_local_index(mc_input)
        self._retrieve_and_track_hash_zch_identities_and_metadata(
            mcm, mc_input, mc_input[table].values()
        )
        return mc_input

    def compute(
        self,
        ctx: ManagedCollisionCollectionContext,
        dist_input: KJTList,
    ) -> KJTList:
        remapped_kjts: List[KeyedJaggedTensor] = []

        # per shard
        for features, sharding_ctx, tables, splits, fns in zip(
            dist_input,
            ctx.sharding_contexts,
            self._sharding_tables,
            self._sharding_per_table_feature_splits,
            self._sharding_features,
        ):
            assert isinstance(sharding_ctx, SequenceShardingContext)
            if not features.variable_stride_per_key():
                # For VBE-Support for EC, EC always pads kjt if VBE in input_dist
                sharding_ctx.lengths_after_input_dist = features.lengths().view(
                    -1, features.stride()
                )

            values: torch.Tensor
            if len(splits) > 1:
                # features per shard split by tables
                feature_splits = features.split(splits)
                output: Dict[str, JaggedTensor] = {}
                for table, kjt in zip(tables, feature_splits):
                    mc_input = self.get_lookup_value(table, kjt)
                    output.update(mc_input)

                values = torch.cat([jt.values() for jt in output.values()])
            else:
                table: str = tables[0]
                mc_input = self.get_lookup_value(table, features)
                values = mc_input[table].values()
            remapped_kjts.append(
                KeyedJaggedTensor(
                    keys=fns,
                    values=values,
                    lengths=features.lengths(),
                    # original weights instead of features splits
                    weights=features.weights_or_none(),
                    stride_per_key_per_rank=features._stride_per_key_per_rank,
                )
            )
        return KJTList(remapped_kjts)

    def evict(self) -> Dict[str, Optional[torch.Tensor]]:
        evictions: Dict[str, Optional[torch.Tensor]] = {}
        for (
            table,
            managed_collision_module,
        ) in self._managed_collision_modules.items():
            # pyrefly: ignore[not-callable]
            global_indices_to_evict = managed_collision_module.evict()
            local_indices_to_evict = None
            if global_indices_to_evict is not None:
                local_indices_to_evict = (
                    global_indices_to_evict - self._table_to_offset[table]
                )
            evictions[table] = local_indices_to_evict
        return evictions

    def open_slots(self) -> Dict[str, torch.Tensor]:
        open_slots: Dict[str, torch.Tensor] = {}
        for (
            table,
            managed_collision_module,
        ) in self._managed_collision_modules.items():
            # pyrefly: ignore[not-callable]
            open_slots[table] = managed_collision_module.open_slots()
        return open_slots

    def output_dist(
        self,
        ctx: ManagedCollisionCollectionContext,
        output: KJTList,
    ) -> LazyAwaitable[KeyedJaggedTensor]:
        global_remapped = self._kjt_list_to_tensor_list(output)
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []

        for odist, remapped_ids, sharding_ctx in zip(
            self._output_dists,
            global_remapped,
            ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(remapped_ids, sharding_ctx))
            features_before_all2all_per_sharding.append(
                #  got `Optional[KeyedJaggedTensor]`.
                # pyrefly: ignore[bad-argument-type]
                sharding_ctx.features_before_input_dist
            )
        return ManagedCollisionCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=False,
            features_to_permute_indices=None,
        )

    def create_context(self) -> ManagedCollisionCollectionContext:
        return ManagedCollisionCollectionContext(sharding_contexts=[])

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for name, module in self._managed_collision_modules.items():
            module_prefix = append_prefix(prefix, name)
            for name, _ in module.named_buffers():
                if name in [
                    "_output_segments_tensor",
                    "_current_iter_tensor",
                    "_scalar_logger._scalar_logger_steps",
                    "_hash_zch_bucket",
                ]:
                    continue
                if name in module._non_persistent_buffers_set:
                    continue
                yield append_prefix(module_prefix, name)
            for name, _ in module.named_parameters():
                yield append_prefix(module_prefix, name)

    @property
    def unsharded_module_type(self) -> Type[ManagedCollisionCollection]:
        return ManagedCollisionCollection

    def register_post_lookup_tracker_fn(
        self,
        record_fn: Callable[
            [
                KeyedJaggedTensor,
                torch.Tensor,
                Optional[nn.Module],
                Optional[torch.Tensor],
                Optional[torch.Tensor],
            ],
            None,
        ],
    ) -> None:
        """
        Register a function to be called after lookup is done. This is used for
        tracking the lookup results and optimizer states.

        Args:
            record_fn (Callable[[KeyedJaggedTensor, torch.Tensor], None]): A custom record function to be called after lookup is done.

        """
        if self.post_lookup_tracker_fn is not None:
            logger.warning(
                "[ModelDeltaTracker] Custom record function already defined, overriding with new callable"
            )
        self.post_lookup_tracker_fn = record_fn


class ManagedCollisionCollectionSharder(
    BaseEmbeddingSharder[ManagedCollisionCollection]
):
    def __init__(
        self,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)

    # pyrefly: ignore[bad-override]
    def shard(
        self,
        module: ManagedCollisionCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ],
        device: Optional[torch.device] = None,
        use_index_dedup: bool = False,
    ) -> ShardedManagedCollisionCollection:

        if device is None:
            device = torch.device("cpu")

        return ShardedManagedCollisionCollection(
            module,
            params,
            env=env,
            device=device,
            embedding_shardings=embedding_shardings,
            use_index_dedup=use_index_dedup,
        )

    def shardable_parameters(
        self, module: ManagedCollisionCollection
    ) -> Dict[str, torch.nn.Parameter]:
        # TODO: standalone sharding
        raise NotImplementedError()

    @property
    def module_type(self) -> Type[ManagedCollisionCollection]:
        return ManagedCollisionCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.ROW_WISE.value,
        ]
        return types


@torch.fx.wrap
def _cat_jagged_values(jd: Dict[str, JaggedTensor]) -> torch.Tensor:
    return torch.cat([jt.values() for jt in jd.values()])


@torch.fx.wrap
def update_jagged_tensor_dict(
    output: Dict[str, JaggedTensor], new_dict: Dict[str, JaggedTensor]
) -> Dict[str, JaggedTensor]:
    output.update(new_dict)
    return output


class ShardedMCCRemapper(nn.Module):
    def __init__(
        self,
        table_feature_splits: List[int],
        fns: List[str],
        managed_collision_modules: nn.ModuleDict,
        shard_metadata: Dict[str, List[int]],
    ) -> None:
        super().__init__()
        self._table_feature_splits: List[int] = table_feature_splits
        self._fns: List[str] = fns
        self.zchs = managed_collision_modules
        logger.info(f"registered zchs: {self.zchs=}")

        # shard_size, shard_offset
        self._shard_metadata: Dict[str, List[int]] = shard_metadata
        self._table_to_offset: Dict[str, int] = {
            table: offset[0] for table, offset in shard_metadata.items()
        }

        # Build feature-to-offset mapping.
        # Features are grouped by table in order: the first _table_feature_splits[0]
        # features in _fns belong to the first table in zchs, the next
        # _table_feature_splits[1] features belong to the second table, etc.
        # This mapping allows downstream code to add table offsets to remapped
        # feature values when returning remapped features for debugging/analysis.
        self._feature_to_offset: Dict[str, int] = {}
        idx = 0
        for i, table in enumerate(self.zchs.keys()):
            offset = self._table_to_offset[table]
            for _ in range(self._table_feature_splits[i]):
                feature_name = self._fns[idx]
                self._feature_to_offset[feature_name] = offset
                idx += 1

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        # features per shard split by tables
        feature_splits = features.split(self._table_feature_splits)
        output: Dict[str, JaggedTensor] = {}
        for i, (table, mc_module) in enumerate(self.zchs.items()):
            kjt: KeyedJaggedTensor = feature_splits[i]
            mc_input: Dict[str, JaggedTensor] = {
                table: JaggedTensor(
                    values=kjt.values(),
                    lengths=kjt.lengths(),
                    weights=_get_length_per_key(kjt),
                )
            }
            remapped_input = mc_module(mc_input)
            mc_input = self.global_to_local_index(remapped_input)
            output[table] = remapped_input[table]

        values: torch.Tensor = _cat_jagged_values(output)
        return KeyedJaggedTensor(
            keys=self._fns,
            values=values,
            lengths=features.lengths(),
            # original weights instead of features splits
            weights=features.weights_or_none(),
        )

    def global_to_local_index(
        self,
        jt_dict: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        return _fx_global_to_local_index(jt_dict, self._table_to_offset)


class ShardedQuantManagedCollisionCollection(
    ShardedModule[
        KJTList,
        KJTList,
        KeyedJaggedTensor,
        ManagedCollisionCollectionContext,
    ]
):
    def __init__(
        self,
        module: ManagedCollisionCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: torch.device,
        embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        # pyrefly: ignore[missing-attribute]
        super().__init__()
        self._env: ShardingEnv = (
            env
            if not isinstance(env, Dict)
            # pyrefly: ignore[missing-attribute]
            else embedding_shardings[0]._env
        )
        self._device = device
        self.need_preprocess: bool = module.need_preprocess
        self._table_name_to_parameter_sharding: Dict[str, ParameterSharding] = (
            copy.deepcopy(table_name_to_parameter_sharding)
        )
        # TODO: create a MCSharding type instead of leveraging EmbeddingSharding
        self._embedding_shardings = embedding_shardings

        self._embedding_names_per_sharding: List[List[str]] = []
        for sharding in self._embedding_shardings:
            # TODO: support TWRW sharding
            assert isinstance(
                sharding, BaseRwEmbeddingSharding
            ), "Only ROW_WISE sharding is supported."
            self._embedding_names_per_sharding.append(sharding.embedding_names())

        self._feature_to_table: Dict[str, str] = module._feature_to_table
        self._table_to_features: Dict[str, List[str]] = module._table_to_features
        self._has_uninitialized_input_dists: bool = True
        self._input_dists: torch.nn.ModuleList = torch.nn.ModuleList([])
        self._managed_collision_modules: nn.ModuleDict = nn.ModuleDict()
        self._create_managed_collision_modules(module)
        self._features_order: List[int] = []

    def _create_managed_collision_modules(
        self, module: ManagedCollisionCollection
    ) -> None:

        self._managed_collision_modules_per_rank: List[torch.nn.ModuleDict] = [
            torch.nn.ModuleDict() for _ in range(self._env.world_size)
        ]
        self._shard_metadata_per_rank: List[Dict[str, List[int]]] = [
            defaultdict() for _ in range(self._env.world_size)
        ]
        self._mc_module_name_shard_metadata: DefaultDict[str, List[int]] = defaultdict()
        # To map mch output indices from local to global. key: table_name
        self._table_to_offset: Dict[str, int] = {}

        # the split sizes of tables belonging to each sharding. outer len is # shardings
        self._sharding_per_table_feature_splits: List[List[int]] = []
        self._input_size_per_table_feature_splits: List[List[int]] = []
        # the split sizes of features per sharding. len is # shardings
        self._sharding_feature_splits: List[int] = []
        # the split sizes of features per table. len is # tables sum over all shardings
        self._table_feature_splits: List[int] = []
        self._feature_names: List[str] = []

        # table names of each sharding
        self._sharding_tables: List[List[str]] = []
        self._sharding_features: List[List[str]] = []

        logger.info(f"_create_managed_collision_modules {self._embedding_shardings=}")

        for sharding in self._embedding_shardings:
            assert isinstance(sharding, BaseRwEmbeddingSharding)
            self._sharding_tables.append([])
            self._sharding_features.append([])
            self._sharding_per_table_feature_splits.append([])
            self._input_size_per_table_feature_splits.append([])

            grouped_embedding_configs: List[GroupedEmbeddingConfig] = (
                sharding._grouped_embedding_configs
            )
            self._sharding_feature_splits.append(len(sharding.feature_names()))

            num_sharding_features = 0
            for group_config in grouped_embedding_configs:
                for table in group_config.embedding_tables:
                    # pyrefly: ignore[missing-attribute]
                    global_meta_data = table.global_metadata.shards_metadata
                    output_segments = [
                        x.shard_offsets[0]
                        # pyrefly: ignore[missing-attribute]
                        for x in table.global_metadata.shards_metadata
                    ] + [table.num_embeddings]
                    mc_module = module._managed_collision_modules[table.name]
                    # pyrefly: ignore[bad-argument-type]
                    mc_module._is_inference = True
                    self._managed_collision_modules[table.name] = mc_module
                    self._sharding_tables[-1].append(table.name)
                    self._sharding_features[-1].extend(table.feature_names)
                    self._feature_names.extend(table.feature_names)
                    logger.info(
                        f"global_meta_data for table {table} is {global_meta_data}"
                    )

                    for i in range(self._env.world_size):
                        new_min_output_id = global_meta_data[i].shard_offsets[0]
                        new_range_size = global_meta_data[i].shard_sizes[0]
                        self._managed_collision_modules_per_rank[i][table.name] = (
                            # pyrefly: ignore[not-callable]
                            mc_module.rebuild_with_output_id_range(
                                output_id_range=(
                                    new_min_output_id,
                                    new_min_output_id + new_range_size,
                                ),
                                output_segments=output_segments,
                                device=(
                                    torch.device("cpu")
                                    if self._device.type == "cpu"
                                    else torch.device(f"{self._device.type}:{i}")
                                ),
                            )
                        )

                        self._managed_collision_modules_per_rank[i][
                            table.name
                        ].training = False
                        self._shard_metadata_per_rank[i][table.name] = [
                            new_min_output_id,
                            new_range_size,
                        ]

                    # pyrefly: ignore[not-callable]
                    input_size = self._managed_collision_modules[
                        table.name
                    ].input_size()

                    self._table_feature_splits.append(len(table.feature_names))
                    self._sharding_per_table_feature_splits[-1].append(
                        self._table_feature_splits[-1]
                    )
                    self._input_size_per_table_feature_splits[-1].append(
                        input_size,
                    )
                    num_sharding_features += self._table_feature_splits[-1]

            assert num_sharding_features == len(
                sharding.feature_names()
            ), f"Shared feature is not supported. {num_sharding_features=}, {self._sharding_per_table_feature_splits[-1]=}"

            if self._sharding_features[-1] != sharding.feature_names():
                logger.warning(
                    "The order of tables of this sharding is altered due to grouping: "
                    f"{self._sharding_features[-1]=} vs {sharding.feature_names()=}"
                )

        logger.info(f"{self._table_feature_splits=}")
        logger.info(f"{self._sharding_per_table_feature_splits=}")
        logger.info(f"{self._input_size_per_table_feature_splits=}")
        logger.info(f"{self._feature_names=}")
        # logger.info(f"{self._table_to_offset=}")
        logger.info(f"{self._sharding_tables=}")
        logger.info(f"{self._sharding_features=}")
        logger.info(f"{self._managed_collision_modules_per_rank=}")
        logger.info(f"{self._shard_metadata_per_rank=}")

    def _create_input_dists(
        self,
        input_feature_names: List[str],
        feature_device: Optional[torch.device] = None,
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._embedding_shardings:
            assert isinstance(sharding, BaseRwEmbeddingSharding)

            emb_sharding = []
            sharding_features = []
            for embedding_table_group in sharding._grouped_embedding_configs_per_rank[
                0
            ]:
                for table in embedding_table_group.embedding_tables:
                    shard_split_offsets = [
                        shard.shard_offsets[0]
                        # pyrefly: ignore[missing-attribute]
                        for shard in table.global_metadata.shards_metadata
                    ]
                    # pyrefly: ignore[missing-attribute]
                    shard_split_offsets.append(table.global_metadata.size[0])
                    emb_sharding.extend(
                        [shard_split_offsets] * len(table.embedding_names)
                    )
                    sharding_features.extend(table.feature_names)

            feature_num_buckets: List[int] = [
                # pyrefly: ignore[not-callable]
                self._managed_collision_modules[self._feature_to_table[f]].buckets()
                for f in sharding_features
            ]

            input_sizes: List[int] = [
                # pyrefly: ignore[not-callable]
                self._managed_collision_modules[self._feature_to_table[f]].input_size()
                for f in sharding_features
            ]

            feature_hash_sizes: List[int] = []
            feature_total_num_buckets: List[int] = []
            for input_size, num_buckets in zip(
                input_sizes,
                feature_num_buckets,
            ):
                feature_hash_sizes.append(input_size)
                feature_total_num_buckets.append(num_buckets)

            input_dist = InferRwSparseFeaturesDist(
                world_size=sharding._world_size,
                num_features=sharding._get_num_features(),
                feature_hash_sizes=feature_hash_sizes,
                feature_total_num_buckets=feature_total_num_buckets,
                device=self._device,
                is_sequence=True,
                has_feature_processor=sharding._has_feature_processor,
                need_pos=False,
                embedding_shard_metadata=emb_sharding,
                keep_original_indices=True,
            )
            self._input_dists.append(input_dist)

            feature_names.extend(sharding_features)

        for f in feature_names:
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(input_feature_names)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(
                self._features_order, device=feature_device, dtype=torch.int32
            ),
            persistent=False,
        )

    # pyrefly: ignore[bad-override]
    def input_dist(
        self,
        ctx: Union[ManagedCollisionCollectionContext, NullShardedModuleContext],
        features: KeyedJaggedTensor,
        is_sequence_embedding: bool = True,
    ) -> ListOfKJTList:
        if self._has_uninitialized_input_dists:
            self._create_input_dists(
                input_feature_names=features.keys(), feature_device=features.device()
            )
            self._has_uninitialized_input_dists = False

        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    # pyrefly: ignore[bad-argument-type]
                    self._features_order_tensor,
                )

            feature_splits: List[KeyedJaggedTensor] = []
            if self.need_preprocess:
                # NOTE: No shared features allowed!
                assert (
                    len(self._sharding_feature_splits) == 1
                ), "Preprocing only support single sharding type (row-wise)"
                table_splits = features.split(self._table_feature_splits)
                ti: int = 0
                for i, tables in enumerate(self._sharding_tables):
                    output: Dict[str, JaggedTensor] = {}
                    for table in tables:
                        kjt: KeyedJaggedTensor = table_splits[ti]
                        mc_module = self._managed_collision_modules[table]
                        # TODO: change to Dict[str, Tensor]
                        mc_input: Dict[str, JaggedTensor] = {
                            table: JaggedTensor(
                                values=kjt.values(),
                                lengths=kjt.lengths(),
                            )
                        }
                        # pyrefly: ignore[not-callable]
                        mc_input = mc_module.preprocess(mc_input)
                        output.update(mc_input)
                        ti += 1
                    shard_kjt = KeyedJaggedTensor(
                        keys=self._sharding_features[i],
                        values=torch.cat([jt.values() for jt in output.values()]),
                        lengths=torch.cat([jt.lengths() for jt in output.values()]),
                    )
                    feature_splits.append(shard_kjt)
            else:
                feature_splits = features.split(self._sharding_feature_splits)

            input_dist_result_list = []
            for feature_split, input_dist in zip(feature_splits, self._input_dists):
                out = input_dist(feature_split)
                input_dist_result_list.append(out.features)
                if is_sequence_embedding:
                    # pyrefly: ignore[missing-attribute]
                    ctx.sharding_contexts.append(
                        InferSequenceShardingContext(
                            features=out.features,
                            features_before_input_dist=features,
                            unbucketize_permute_tensor=(
                                out.unbucketize_permute_tensor
                                if isinstance(input_dist, InferRwSparseFeaturesDist)
                                else None
                            ),
                            bucket_mapping_tensor=out.bucket_mapping_tensor,
                            bucketized_length=out.bucketized_length,
                        )
                    )

        return ListOfKJTList(input_dist_result_list)

    def create_mcc_remappers(self) -> List[List[ShardedMCCRemapper]]:
        ret: List[List[ShardedMCCRemapper]] = []
        # per shard
        for table_feature_splits, fns in zip(
            self._sharding_per_table_feature_splits,
            self._sharding_features,
        ):
            sharding_ret: List[ShardedMCCRemapper] = []
            for i, mcms in enumerate(self._managed_collision_modules_per_rank):
                sharding_ret.append(
                    ShardedMCCRemapper(
                        table_feature_splits=table_feature_splits,
                        fns=fns,
                        managed_collision_modules=mcms,
                        shard_metadata=self._shard_metadata_per_rank[i],
                    )
                )
            ret.append(sharding_ret)
        return ret

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def compute(
        self,
        ctx: ManagedCollisionCollectionContext,
        rank: int,
        dist_input: KJTList,
    ) -> KJTList:
        raise NotImplementedError()

    # pyrefly: ignore[bad-override]
    def output_dist(
        self,
        ctx: ManagedCollisionCollectionContext,
        output: KJTList,
    ) -> KeyedJaggedTensor:
        # For quant/inference, return the remapped features directly
        # without using awaitables (unlike the training version)
        if len(output) == 0:
            return KeyedJaggedTensor.empty()

        if len(output) == 1:
            return output[0]

        # Multiple ranks - combine values and unbucketize to align with input KJT
        # Use self._feature_names instead of output[0].keys() for torch.fx tracing
        # compatibility (Proxy objects cannot be iterated)
        keys: List[str] = self._feature_names

        # Concatenate values from all ranks
        all_values: List[torch.Tensor] = [kjt.values() for kjt in output]
        combined_values = torch.cat(all_values) if all_values else torch.empty(0)

        # Get unbucketize tensor from sharding context to reorder values
        # back to original input order
        unbucketize_tensor: Optional[torch.Tensor] = None
        original_lengths: Optional[torch.Tensor] = None
        if ctx.sharding_contexts:
            sharding_ctx = ctx.sharding_contexts[0]
            if hasattr(sharding_ctx, "unbucketize_permute_tensor"):
                unbucketize_tensor = sharding_ctx.unbucketize_permute_tensor
            if hasattr(sharding_ctx, "features_before_input_dist"):
                features_before = sharding_ctx.features_before_input_dist
                if features_before is not None:
                    original_lengths = features_before.lengths()

        if unbucketize_tensor is not None:
            # Reorder values back to original input order
            combined_values = combined_values.index_select(0, unbucketize_tensor)

        # Use original lengths if available, otherwise concatenate
        if original_lengths is not None:
            lengths = original_lengths
        else:
            all_lengths: List[torch.Tensor] = [kjt.lengths() for kjt in output]
            lengths = (
                torch.cat(all_lengths)
                if all_lengths
                else torch.empty(0, dtype=torch.int32)
            )

        return KeyedJaggedTensor(
            keys=keys,
            values=combined_values,
            lengths=lengths,
        )

    def create_context(self) -> ManagedCollisionCollectionContext:
        return ManagedCollisionCollectionContext(sharding_contexts=[])

    @property
    def unsharded_module_type(self) -> Type[ManagedCollisionCollection]:
        return ManagedCollisionCollection


class InferManagedCollisionCollectionSharder(ManagedCollisionCollectionSharder):
    # pyrefly: ignore[bad-override]
    def shard(
        self,
        module: ManagedCollisionCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ],
        device: Optional[torch.device] = None,
    ) -> ShardedQuantManagedCollisionCollection:

        if device is None:
            device = torch.device("cpu")

        return ShardedQuantManagedCollisionCollection(
            module,
            params,
            env=env,
            device=device,
            embedding_shardings=embedding_shardings,
        )
