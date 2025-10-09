#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torchrec.distributed.embedding import (
    create_sharding_infos_by_sharding_device_group,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_sharding import EmbeddingSharding
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    EmbeddingComputeKernel,
    FeatureShardingMixIn,
    GroupedEmbeddingConfig,
    InputDistOutputs,
    KJTList,
    ListOfKJTList,
    ShardingType,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_IS_SSD_TABLE,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
    get_tbes_to_register_from_iterable,
    is_fused_param_quant_state_dict_split_scale_bias,
    is_fused_param_register_tbe,
)
from torchrec.distributed.global_settings import get_propogate_device
from torchrec.distributed.mc_modules import (
    InferManagedCollisionCollectionSharder,
    ShardedMCCRemapper,
    ShardedQuantManagedCollisionCollection,
)
from torchrec.distributed.quant_state import ShardedQuantEmbeddingModuleState
from torchrec.distributed.sharding.cw_sequence_sharding import (
    InferCwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    InferRwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.sequence_sharding import (
    InferSequenceShardingContext,
    SequenceShardingContext,
)
from torchrec.distributed.sharding.tw_sequence_sharding import (
    InferTwSequenceEmbeddingSharding,
)
from torchrec.distributed.types import ParameterSharding, ShardingEnv, ShardMetadata
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingConfig,
)
from torchrec.modules.utils import (
    _fx_trec_get_feature_length,
    _get_batching_hinted_output,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    QuantManagedCollisionEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable

torch.fx.wrap("len")
torch.fx.wrap("_get_batching_hinted_output")
torch.fx.wrap("_fx_trec_get_feature_length")

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except (OSError, RuntimeError):
    pass


logger: logging.Logger = logging.getLogger(__name__)


ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


@dataclass
class EmbeddingCollectionContext(Multistreamable):
    sharding_contexts: List[InferSequenceShardingContext]

    def record_stream(self, stream: torch.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)


class ManagedCollisionEmbeddingCollectionContext(EmbeddingCollectionContext):

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        evictions_per_table: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        remapped_kjt: Optional[KJTList] = None,
    ) -> None:
        # pyre-ignore
        super().__init__(sharding_contexts)
        self.evictions_per_table: Optional[Dict[str, Optional[torch.Tensor]]] = (
            evictions_per_table
        )
        self.remapped_kjt: Optional[KJTList] = remapped_kjt

    def record_stream(self, stream: torch.Stream) -> None:
        super().record_stream(stream)
        if self.evictions_per_table:
            #  pyre-ignore
            for value in self.evictions_per_table.values():
                if value is None:
                    continue
                value.record_stream(stream)
        if self.remapped_kjt is not None:
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            self.remapped_kjt.record_stream(stream)


def get_device_from_parameter_sharding(
    ps: ParameterSharding,
) -> Union[str, Tuple[str, ...]]:
    """
    Returns list ofdevice type / shard if table is sharded across different device type
    else reutrns single device type for the table parameter
    """
    if not isinstance(ps.sharding_spec, EnumerableShardingSpec):
        raise ValueError("Expected EnumerableShardingSpec as input to the function")

    device_type_list: Tuple[str, ...] = tuple(
        # pyre-fixme[16]: `Optional` has no attribute `device`
        [shard.placement.device().type for shard in ps.sharding_spec.shards]
    )
    if len(set(device_type_list)) == 1:
        return device_type_list[0]
    else:
        assert ps.sharding_type in [
            ShardingType.ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ], "Only row_wise or column_wise sharding supports sharding across multiple device types for a table"
        return device_type_list


def get_device_from_sharding_infos(
    emb_shard_infos: List[EmbeddingShardingInfo],
) -> Union[str, Tuple[str, ...]]:
    res_set = set()
    for emb_shard_info in emb_shard_infos:
        if emb_shard_info.fused_params and emb_shard_info.fused_params.get(
            FUSED_PARAM_IS_SSD_TABLE, False
        ):
            res_set.add("ssd")
        else:
            res_set.add(
                get_device_from_parameter_sharding(emb_shard_info.param_sharding)
            )
    res = list(res_set)
    assert len(res) == 1, "All shards should be on the same type of device"
    return res[0]


def get_device_for_first_shard_from_sharding_infos(
    emb_shard_infos: List[EmbeddingShardingInfo],
) -> str:
    device_type = get_device_from_sharding_infos(emb_shard_infos)
    return device_type[0] if isinstance(device_type, tuple) else device_type


def create_infer_embedding_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[
    InferSequenceShardingContext,
    InputDistOutputs,
    List[torch.Tensor],
    List[torch.Tensor],
]:
    storage_device_type_from_sharding_infos: Union[str, Tuple[str, ...]] = (
        get_device_from_sharding_infos(sharding_infos)
    )

    if storage_device_type_from_sharding_infos in ["cuda", "mtia"]:
        if sharding_type == ShardingType.TABLE_WISE.value:
            assert isinstance(
                storage_device_type_from_sharding_infos, str
            ), "embedding storage device type should always be a str for tw sharding and not a tuple"
            return InferTwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                storage_device_type_from_sharding_infos=storage_device_type_from_sharding_infos,
            )
        elif sharding_type == ShardingType.COLUMN_WISE.value:
            return InferCwSequenceEmbeddingSharding(sharding_infos, env, device)
        elif sharding_type == ShardingType.ROW_WISE.value:
            return InferRwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                device_type_from_sharding_infos=storage_device_type_from_sharding_infos,
            )
        else:
            raise ValueError(
                f"Sharding type not supported {sharding_type} for {storage_device_type_from_sharding_infos} sharding"
            )
    elif storage_device_type_from_sharding_infos in ["cpu", "ssd"] or isinstance(
        storage_device_type_from_sharding_infos, tuple
    ):
        if sharding_type == ShardingType.ROW_WISE.value:
            return InferRwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                device_type_from_sharding_infos=storage_device_type_from_sharding_infos,
            )
        elif sharding_type == ShardingType.TABLE_WISE.value:
            assert isinstance(
                storage_device_type_from_sharding_infos, str
            ), "embedding storage device type should always be a str for tw sharding and not a tuple"
            return InferTwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                storage_device_type_from_sharding_infos=storage_device_type_from_sharding_infos,
            )
        else:
            raise ValueError(
                f"Sharding type not supported {sharding_type} for {storage_device_type_from_sharding_infos} sharding"
            )
    else:
        raise ValueError(
            f"Sharding type not supported {sharding_type} for {storage_device_type_from_sharding_infos} sharding"
        )


@torch.fx.wrap
def _fx_trec_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def _fx_trec_wrap_length_tolist(length: torch.Tensor) -> List[int]:
    return length.long().tolist()


@torch.fx.wrap
def _get_unbucketize_tensor_via_length_alignment(
    lengths: torch.Tensor,
    bucketize_length: torch.Tensor,
    bucketize_permute_tensor: torch.Tensor,
    bucket_mapping_tensor: torch.Tensor,
) -> torch.Tensor:
    return bucketize_permute_tensor


def _construct_jagged_tensors_tw(
    embeddings: List[torch.Tensor],
    embedding_names_per_rank: List[List[str]],
    features: KJTList,
    need_indices: bool,
    storage_device_type: str,
) -> Dict[str, JaggedTensor]:
    ret: Dict[str, JaggedTensor] = {}
    for i in range(len(embedding_names_per_rank)):
        embeddings_i = embeddings[i]
        features_i: KeyedJaggedTensor = features[i]
        if storage_device_type in ["ssd", "cpu"]:
            embeddings_i = _get_batching_hinted_output(
                _fx_trec_get_feature_length(features_i, embedding_names_per_rank[i]),
                embeddings_i,
            )

        lengths = features_i.lengths().view(-1, features_i.stride())
        values = features_i.values()
        length_per_key = features_i.length_per_key()

        embeddings_list = torch.split(embeddings_i, length_per_key, dim=0)
        stride = features_i.stride()
        lengths_tuple = torch.unbind(lengths.view(-1, stride), dim=0)
        if need_indices:
            values_list = torch.split(values, length_per_key)
            for j, key in enumerate(embedding_names_per_rank[i]):
                ret[key] = JaggedTensor(
                    lengths=lengths_tuple[j],
                    values=embeddings_list[j],
                    weights=values_list[j],
                )
        else:
            for j, key in enumerate(embedding_names_per_rank[i]):
                ret[key] = JaggedTensor(
                    lengths=lengths_tuple[j],
                    values=embeddings_list[j],
                    weights=None,
                )
    return ret


@torch.fx.wrap
def _fx_marker_construct_jagged_tensor(
    values: torch.Tensor,
    lengths: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> JaggedTensor:
    return JaggedTensor(values=values, lengths=lengths, weights=weights)


def _construct_jagged_tensors_rw(
    embeddings: List[torch.Tensor],
    feature_keys: List[str],
    feature_length: torch.Tensor,
    feature_indices: Optional[torch.Tensor],
    need_indices: bool,
    unbucketize_tensor: torch.Tensor,
) -> Dict[str, JaggedTensor]:
    ret: Dict[str, JaggedTensor] = {}
    unbucketized_embs = torch.concat(embeddings, dim=0).index_select(
        0, unbucketize_tensor
    )
    feature_length_2d = feature_length.view(len(feature_keys), -1)
    length_per_key: List[int] = _fx_trec_wrap_length_tolist(
        torch.sum(feature_length_2d, dim=1)
    )
    embs_split_per_key = unbucketized_embs.split(length_per_key, dim=0)
    lengths_list = torch.unbind(feature_length_2d, dim=0)
    values_list: List[torch.Tensor] = []
    if need_indices:
        # pyre-ignore
        values_list = torch.split(
            _fx_trec_unwrap_optional_tensor(feature_indices),
            length_per_key,
        )
    for i, key in enumerate(feature_keys):
        ret[key] = _fx_marker_construct_jagged_tensor(
            values=embs_split_per_key[i],
            lengths=lengths_list[i],
            weights=values_list[i] if need_indices else None,
        )
    return ret


@torch.fx.wrap
def _construct_jagged_tensors_cw(
    embeddings: List[torch.Tensor],
    features: KJTList,
    embedding_names_per_rank: List[List[str]],
    need_indices: bool,
    features_to_permute_indices: Dict[str, torch.Tensor],
    key_to_feature_permuted_coordinates: Dict[str, torch.Tensor],
) -> Dict[str, JaggedTensor]:
    ret: Dict[str, JaggedTensor] = {}
    stride = features[0].stride()
    lengths_lists: List[List[torch.Tensor]] = []
    embeddings_lists: List[List[torch.Tensor]] = []
    values_lists: List[List[torch.Tensor]] = []
    for i in range(len(features)):
        embedding = embeddings[i]
        feature = features[i]
        # pyre-fixme[6]: For 1st argument expected `List[Tensor]` but got
        #  `Tuple[Tensor, ...]`.
        lengths_lists.append(torch.unbind(feature.lengths().view(-1, stride), dim=0))
        embeddings_lists.append(
            list(torch.split(embedding, feature.length_per_key(), dim=0))
        )
    if need_indices:
        for i in range(len(features)):
            feature = features[i]
            values_lists.append(
                list(torch.split(feature.values(), feature.length_per_key()))
            )

    for key, permuted_coordinate_tensor in key_to_feature_permuted_coordinates.items():
        permuted_coordinates: List[List[int]] = permuted_coordinate_tensor.tolist()

        rank0, idx_in_rank0 = permuted_coordinates[0]
        ret[key] = JaggedTensor(
            lengths=lengths_lists[rank0][idx_in_rank0],
            values=torch.cat(
                [
                    embeddings_lists[rank][idx_in_rank]
                    for rank, idx_in_rank in permuted_coordinates
                ],
                dim=1,
            ),
            weights=values_lists[rank0][idx_in_rank0] if need_indices else None,
        )
    return ret


@torch.fx.wrap
def input_dist_permute(
    features: KeyedJaggedTensor,
    features_order: List[int],
    features_order_tensor: torch.Tensor,
) -> KeyedJaggedTensor:
    return features.permute(
        features_order,
        features_order_tensor,
    )


def _construct_jagged_tensors(
    sharding_type: str,
    embeddings: List[torch.Tensor],
    features: KJTList,
    embedding_names: List[str],
    embedding_names_per_rank: List[List[str]],
    features_before_input_dist: KeyedJaggedTensor,
    need_indices: bool,
    rw_unbucketize_tensor: Optional[torch.Tensor],
    rw_bucket_mapping_tensor: Optional[torch.Tensor],
    rw_feature_length_after_bucketize: Optional[torch.Tensor],
    cw_features_to_permute_indices: Dict[str, torch.Tensor],
    key_to_feature_permuted_coordinates: Dict[str, torch.Tensor],
    device_type: Union[str, Tuple[str, ...]],
) -> Dict[str, JaggedTensor]:

    # Validating sharding type and parameters
    valid_sharding_types = [
        ShardingType.ROW_WISE.value,
        ShardingType.COLUMN_WISE.value,
        ShardingType.TABLE_WISE.value,
    ]
    if sharding_type not in valid_sharding_types:
        raise ValueError(f"Unknown sharding type {sharding_type}")

    if sharding_type == ShardingType.ROW_WISE.value and rw_unbucketize_tensor is None:
        raise ValueError("rw_unbucketize_tensor is required for row-wise sharding")

    if sharding_type == ShardingType.ROW_WISE.value:
        features_before_input_dist_length = _fx_trec_get_feature_length(
            features_before_input_dist, embedding_names
        )
        input_embeddings = []
        for i in range(len(embedding_names_per_rank)):
            if isinstance(device_type, tuple) and device_type[i] != "cpu":
                # batching hint is already propagated and passed for this case
                # upstream
                input_embeddings.append(embeddings[i])
            else:
                input_embeddings.append(
                    _get_batching_hinted_output(
                        _fx_trec_get_feature_length(
                            features[i], embedding_names_per_rank[i]
                        ),
                        embeddings[i],
                    )
                )

        return _construct_jagged_tensors_rw(
            input_embeddings,
            embedding_names,
            features_before_input_dist_length,
            features_before_input_dist.values() if need_indices else None,
            need_indices,
            _get_unbucketize_tensor_via_length_alignment(
                features_before_input_dist_length,
                rw_feature_length_after_bucketize,
                rw_unbucketize_tensor,
                rw_bucket_mapping_tensor,
            ),
        )

    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return _construct_jagged_tensors_cw(
            embeddings,
            features,
            embedding_names_per_rank,
            need_indices,
            cw_features_to_permute_indices,
            key_to_feature_permuted_coordinates,
        )
    else:  # sharding_type == ShardingType.TABLE_WISE.value
        assert isinstance(
            device_type, str
        ), "device_type should be a str for tw sharding and not a tuple"
        return _construct_jagged_tensors_tw(
            embeddings,
            embedding_names_per_rank,
            features,
            need_indices,
            str(device_type),
        )


# Wrap the annotation in a separate function with input parameter so that it won't be dropped during symbolic trace.
# Please note the input parameter is necessary, though is not used, otherwise this function will be optimized.
@torch.fx.has_side_effect
@torch.fx.wrap
def annotate_embedding_names(
    embedding_names: List[str],
    dummy: List[List[torch.Tensor]],
) -> List[str]:
    return torch.jit.annotate(List[str], embedding_names)


@torch.fx.wrap
def format_embedding_names_per_rank_per_sharding(
    embedding_names_per_rank_per_sharding: List[List[List[str]]],
    dummy: List[List[torch.Tensor]],
) -> List[List[List[str]]]:
    annotated_embedding_names_per_rank_per_sharding: List[List[List[str]]] = []
    for embedding_names_per_rank in embedding_names_per_rank_per_sharding:
        annotated_embedding_names_per_rank: List[List[str]] = []
        for embedding_names in embedding_names_per_rank:
            annotated_embedding_names_per_rank.append(
                annotate_embedding_names(embedding_names, dummy)
            )
        annotated_embedding_names_per_rank_per_sharding.append(
            annotated_embedding_names_per_rank
        )
    return annotated_embedding_names_per_rank_per_sharding


class ShardedQuantEmbeddingCollection(
    ShardedQuantEmbeddingModuleState[
        ListOfKJTList,
        List[List[torch.Tensor]],
        Dict[str, JaggedTensor],
        EmbeddingCollectionContext,
    ],
):
    """
    Sharded implementation of `QuantEmbeddingCollection`.
    """

    def __init__(
        self,
        module: QuantEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        # TODO: Consolidate to use Dict[str, ShardingEnv]
        env: Union[
            ShardingEnv, Dict[str, ShardingEnv]
        ],  # Support hybrid sharding for DI
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self._embedding_configs: List[EmbeddingConfig] = module.embedding_configs()

        self._sharding_type_device_group_to_sharding_infos: Dict[
            Tuple[str, Union[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]
        ] = create_sharding_infos_by_sharding_device_group(
            module, table_name_to_parameter_sharding, fused_params
        )

        self._sharding_type_device_group_to_sharding: Dict[
            Tuple[str, Union[str, Tuple[str, ...]]],
            EmbeddingSharding[
                InferSequenceShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                List[torch.Tensor],
            ],
        ] = {
            (sharding_type, device_group): create_infer_embedding_sharding(
                sharding_type,
                embedding_configs,
                (
                    env
                    if not isinstance(env, Dict)
                    else env[
                        get_device_for_first_shard_from_sharding_infos(
                            embedding_configs
                        )
                    ]
                ),
                device if get_propogate_device() else None,
            )
            for (
                sharding_type,
                device_group,
            ), embedding_configs in self._sharding_type_device_group_to_sharding_infos.items()
        }
        self._embedding_dim: int = module.embedding_dim()
        self._local_embedding_dim: int = self._embedding_dim
        self._all_embedding_names: Set[str] = set()
        self._embedding_names_per_sharding: List[List[str]] = []
        self._embedding_names_per_rank_per_sharding: List[List[List[str]]] = []
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._embedding_names_per_sharding.append(sharding.embedding_names())
            self._all_embedding_names.update(sharding.embedding_names())
            self._embedding_names_per_rank_per_sharding.append(
                sharding.embedding_names_per_rank()
            )
        self._features_to_permute_indices: Dict[str, torch.Tensor] = {}
        self._key_to_feature_permuted_coordinates_per_sharding: List[
            Dict[str, torch.Tensor]
        ] = [{} for i in range(len(self._embedding_names_per_rank_per_sharding))]

        for (
            sharding_type,
            device_group,
        ) in self._sharding_type_device_group_to_sharding.keys():
            if sharding_type == ShardingType.COLUMN_WISE.value:
                sharding = self._sharding_type_device_group_to_sharding[
                    (sharding_type, device_group)
                ]
                # CW partition must be same for all CW sharded parameters
                self._local_embedding_dim = cast(
                    ShardMetadata, sharding.embedding_shard_metadata()[0]
                ).shard_sizes[1]
                self._features_to_permute_indices = (
                    self._generate_permute_indices_per_feature(
                        module.embedding_configs(), table_name_to_parameter_sharding
                    )
                )

                self._generate_permute_coordinates_per_feature_per_sharding()

        self._device = device
        self._lookups: List[nn.Module] = []
        self._create_lookups(fused_params, device)

        # Ensure output dist is set for post processing from an inference runtime (ie. setting device from runtime).
        self._output_dists: torch.nn.ModuleList = torch.nn.ModuleList()

        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_output_dist: bool = True

        self._embedding_dim: int = module.embedding_dim()
        self._need_indices: bool = module.need_indices()

        self._fused_params = fused_params

        tbes: Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig] = (
            get_tbes_to_register_from_iterable(self._lookups)
        )

        self._tbes_configs: Dict[
            IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig
        ] = tbes

        # Optional registration of TBEs for model post processing utilities
        if is_fused_param_register_tbe(fused_params):
            self.tbes: torch.nn.ModuleList = torch.nn.ModuleList(tbes.keys())

        quant_state_dict_split_scale_bias = (
            is_fused_param_quant_state_dict_split_scale_bias(fused_params)
        )

        if quant_state_dict_split_scale_bias:
            self._initialize_torch_state(
                tbes=tbes,
                table_name_to_parameter_sharding=table_name_to_parameter_sharding,
                tables_weights_prefix="embeddings",
            )
        else:
            assert not isinstance(
                env, Dict
            ), "CPU sharding currently only support RW sharding where split scale and bias is required"

            table_wise_sharded_only: bool = all(
                sharding_type == ShardingType.TABLE_WISE.value
                for (
                    sharding_type,
                    _,
                ) in self._sharding_type_device_group_to_sharding.keys()
            )
            assert (
                table_wise_sharded_only
            ), "ROW_WISE,COLUMN_WISE shardings can be used only in 'quant_state_dict_split_scale_bias' mode, specify fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS]=True to __init__ argument"

            self.embeddings: nn.ModuleDict = nn.ModuleDict()
            for table in self._embedding_configs:
                self.embeddings[table.name] = torch.nn.Module()

            for _sharding_type, lookup in zip(
                self._sharding_type_device_group_to_sharding.keys(), self._lookups
            ):
                lookup_state_dict = lookup.state_dict()
                for key in lookup_state_dict:
                    if key.endswith(".weight"):
                        table_name = key[: -len(".weight")]
                        self.embeddings[table_name].register_buffer(
                            "weight", lookup_state_dict[key]
                        )

    def tbes_configs(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return self._tbes_configs

    def sharding_type_device_group_to_sharding_infos(
        self,
    ) -> Dict[Tuple[str, Union[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]]:
        return self._sharding_type_device_group_to_sharding_infos

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_configs

    def _generate_permute_indices_per_feature(
        self,
        embedding_configs: List[EmbeddingConfig],
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    ) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = {}
        shared_feature: Dict[str, bool] = {}
        for table in embedding_configs:
            for feature_name in table.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        for table in embedding_configs:
            sharding = table_name_to_parameter_sharding[table.name]
            if sharding.sharding_type != ShardingType.COLUMN_WISE.value:
                continue
            ranks = cast(List[int], sharding.ranks)
            rank_to_indices = defaultdict(deque)
            for i, rank in enumerate(sorted(ranks)):
                rank_to_indices[rank].append(i)
            permute_indices = [rank_to_indices[rank].popleft() for rank in ranks]
            tensor = torch.tensor(permute_indices, dtype=torch.int64)
            for feature_name in table.feature_names:
                if shared_feature[feature_name]:
                    ret[feature_name + "@" + table.name] = tensor
                else:
                    ret[feature_name] = tensor
        return ret

    def _generate_permute_coordinates_per_feature_per_sharding(
        self,
    ) -> None:
        key_to_feature_permuted_coordinates_per_sharding: List[
            Dict[str, List[Tuple[int, int]]]
        ] = [{} for i in range(len(self._embedding_names_per_rank_per_sharding))]

        for idx, embedding_names_per_rank in enumerate(
            self._embedding_names_per_rank_per_sharding
        ):
            for rank, embedding_names in enumerate(embedding_names_per_rank):
                for idx_in_rank, embedding_name in enumerate(embedding_names):
                    if (
                        embedding_name
                        not in key_to_feature_permuted_coordinates_per_sharding[idx]
                    ):
                        key_to_feature_permuted_coordinates_per_sharding[idx][
                            embedding_name
                        ] = torch.jit.annotate(List[Tuple[int, int]], [])
                    key_to_feature_permuted_coordinates_per_sharding[idx][
                        embedding_name
                    ].append((rank, idx_in_rank))

            for (
                key,
                coordinates,
            ) in key_to_feature_permuted_coordinates_per_sharding[idx].items():
                permuted_coordinates: List[Tuple[int, int]] = coordinates

                if key in self._features_to_permute_indices:
                    permuted_coordinates = [(-1, -1)] * len(coordinates)
                    permute_indices: List[int] = self._features_to_permute_indices[
                        key
                    ].tolist()
                    for i, permute_idx in enumerate(permute_indices):
                        permuted_coordinates[i] = coordinates[permute_idx]
                self._key_to_feature_permuted_coordinates_per_sharding[idx][key] = (
                    torch.tensor(permuted_coordinates)
                )

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._lookups.append(
                sharding.create_lookup(fused_params=fused_params, device=device)
            )

    def _create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))

    # pyre-ignore [14]
    # pyre-ignore
    def input_dist(
        self,
        ctx: EmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> ListOfKJTList:
        if self._has_uninitialized_input_dist:
            # pyre-fixme[16]: `ShardedQuantEmbeddingCollection` has no attribute
            #  `_input_dist`.
            self._input_dist = ShardedQuantEcInputDist(
                input_feature_names=features.keys() if features is not None else [],
                sharding_type_device_group_to_sharding=self._sharding_type_device_group_to_sharding,
                device=self._device,
                feature_device=features.device(),
            )
            self._has_uninitialized_input_dist = False
        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False

        (
            input_dist_result_list,
            features_by_sharding,
            unbucketize_permute_tensor_list,
            bucket_mapping_tensor_list,
            bucketized_length_list,
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        ) = self._input_dist(features)

        with torch.no_grad():
            for i in range(len(self._sharding_type_device_group_to_sharding)):

                ctx.sharding_contexts.append(
                    InferSequenceShardingContext(
                        features=input_dist_result_list[i],
                        features_before_input_dist=features_by_sharding[i],
                        unbucketize_permute_tensor=unbucketize_permute_tensor_list[i],
                        bucket_mapping_tensor=bucket_mapping_tensor_list[i],
                        bucketized_length=bucketized_length_list[i],
                        embedding_names_per_rank=self._embedding_names_per_rank_per_sharding[
                            i
                        ],
                    )
                )
        return input_dist_result_list

    def _embedding_dim_for_sharding_type(self, sharding_type: str) -> int:
        return (
            self._local_embedding_dim
            if sharding_type == ShardingType.COLUMN_WISE.value
            else self._embedding_dim
        )

    def compute(
        self, ctx: EmbeddingCollectionContext, dist_input: ListOfKJTList
    ) -> List[List[torch.Tensor]]:
        ret: List[List[torch.Tensor]] = []

        # for lookup, features in zip(self._lookups, dist_input):
        for i in range(len(self._lookups)):
            lookup = self._lookups[i]
            features = dist_input[i]
            ret.append(lookup.forward(features))
        return ret

    # pyre-ignore
    def output_dist(
        self, ctx: EmbeddingCollectionContext, output: List[List[torch.Tensor]]
    ) -> Dict[str, JaggedTensor]:
        emb_per_sharding: List[List[torch.Tensor]] = []
        features_before_input_dist_per_sharding: List[KeyedJaggedTensor] = []
        features_per_sharding: List[KJTList] = []
        unbucketize_tensors: List[Optional[torch.Tensor]] = []
        bucket_mapping_tensors: List[Optional[torch.Tensor]] = []
        bucketized_lengths: List[Optional[torch.Tensor]] = []
        for sharding_output_dist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
            ctx.sharding_contexts,
        ):
            sharding_output_dist_res: List[torch.Tensor] = sharding_output_dist.forward(
                embeddings, sharding_ctx
            )
            emb_per_sharding.append(sharding_output_dist_res)
            features_per_sharding.append(sharding_ctx.features)
            unbucketize_tensors.append(
                sharding_ctx.unbucketize_permute_tensor
                if sharding_ctx.unbucketize_permute_tensor is not None
                else None
            )
            bucket_mapping_tensors.append(
                sharding_ctx.bucket_mapping_tensor
                if sharding_ctx.bucket_mapping_tensor is not None
                else None
            )
            bucketized_lengths.append(
                sharding_ctx.bucketized_length
                if sharding_ctx.bucketized_length is not None
                else None
            )
            features_before_input_dist_per_sharding.append(
                # pyre-ignore
                sharding_ctx.features_before_input_dist
            )
        return self.output_jt_dict(
            emb_per_sharding=emb_per_sharding,
            features_per_sharding=features_per_sharding,
            features_before_input_dist_per_sharding=features_before_input_dist_per_sharding,
            unbucketize_tensors=unbucketize_tensors,
            bucket_mapping_tensors=bucket_mapping_tensors,
            bucketized_lengths=bucketized_lengths,
        )

    def output_jt_dict(
        self,
        emb_per_sharding: List[List[torch.Tensor]],
        features_per_sharding: List[KJTList],
        features_before_input_dist_per_sharding: List[KeyedJaggedTensor],
        unbucketize_tensors: List[Optional[torch.Tensor]],
        bucket_mapping_tensors: List[Optional[torch.Tensor]],
        bucketized_lengths: List[Optional[torch.Tensor]],
    ) -> Dict[str, JaggedTensor]:
        jt_dict_res: Dict[str, JaggedTensor] = {}
        for (
            (sharding_type, device_type),
            emb_sharding,
            features_sharding,
            embedding_names,
            embedding_names_per_rank,
            features_before_input_dist,
            unbucketize_tensor,
            bucket_mapping_tensor,
            bucketized_length,
            key_to_feature_permuted_coordinates,
        ) in zip(
            self._sharding_type_device_group_to_sharding.keys(),
            emb_per_sharding,
            features_per_sharding,
            self._embedding_names_per_sharding,
            self._embedding_names_per_rank_per_sharding,
            features_before_input_dist_per_sharding,
            unbucketize_tensors,
            bucket_mapping_tensors,
            bucketized_lengths,
            self._key_to_feature_permuted_coordinates_per_sharding,
        ):
            jt_dict = _construct_jagged_tensors(
                sharding_type=sharding_type,
                embeddings=emb_sharding,
                features=features_sharding,
                embedding_names=embedding_names,
                embedding_names_per_rank=embedding_names_per_rank,
                features_before_input_dist=features_before_input_dist,
                need_indices=self._need_indices,
                rw_unbucketize_tensor=(
                    # this is batching hint for constructing alignment sparse features for batching
                    _fx_trec_unwrap_optional_tensor(unbucketize_tensor)
                    if sharding_type == ShardingType.ROW_WISE.value
                    else None
                ),
                rw_bucket_mapping_tensor=(
                    _fx_trec_unwrap_optional_tensor(bucket_mapping_tensor)
                    if sharding_type == ShardingType.ROW_WISE.value
                    else None
                ),
                rw_feature_length_after_bucketize=(
                    _fx_trec_unwrap_optional_tensor(bucketized_length)
                    if sharding_type == ShardingType.ROW_WISE.value
                    else None
                ),
                cw_features_to_permute_indices=self._features_to_permute_indices,
                key_to_feature_permuted_coordinates=key_to_feature_permuted_coordinates,
                device_type=device_type,
            )
            for embedding_name in embedding_names:
                jt_dict_res[embedding_name] = jt_dict[embedding_name]

        return jt_dict_res

    # pyre-ignore
    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, input: ListOfKJTList
    ) -> Dict[str, JaggedTensor]:
        return self.output_dist(ctx, self.compute(ctx, input))

    # pyre-ignore
    def forward(self, *input, **kwargs) -> Dict[str, JaggedTensor]:
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input, **kwargs)
        return self.compute_and_output_dist(ctx, dist_input)

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return super().copy(device)

    def create_context(self) -> EmbeddingCollectionContext:
        return EmbeddingCollectionContext(sharding_contexts=[])

    @property
    def shardings(
        self,
    ) -> Dict[Tuple[str, Union[str, Tuple[str, ...]]], FeatureShardingMixIn]:
        # pyre-ignore [7]
        return self._sharding_type_device_group_to_sharding


class QuantEmbeddingCollectionSharder(
    BaseQuantEmbeddingSharder[QuantEmbeddingCollection]
):
    """
    This implementation uses non-fused EmbeddingCollection
    """

    def shard(
        self,
        module: QuantEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS not in fused_params:
            fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
                module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            )
        if FUSED_PARAM_REGISTER_TBE_BOOL not in fused_params:
            fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
                module, FUSED_PARAM_REGISTER_TBE_BOOL, False
            )
        return ShardedQuantEmbeddingCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=fused_params,
            device=device,
        )

    @property
    def module_type(self) -> Type[QuantEmbeddingCollection]:
        return QuantEmbeddingCollection


class ShardedQuantEcInputDist(torch.nn.Module):
    """
    This module implements distributed inputs of a ShardedQuantEmbeddingCollection.

    Args:
        input_feature_names (List[str]): EmbeddingCollection feature names.
        sharding_type_to_sharding (Dict[
            str,
            EmbeddingSharding[
                InferSequenceShardingContext,
                KJTList,
                List[torch.Tensor],
                List[torch.Tensor],
            ],
        ]): map from sharding type to EmbeddingSharding.
        device (Optional[torch.device]): default compute device.
        feature_device (Optional[torch.device]): runtime feature device.

    Example::

        sqec_input_dist = ShardedQuantEcInputDist(
            sharding_type_to_sharding={
                ShardingType.TABLE_WISE: InferTwSequenceEmbeddingSharding(
                    [],
                    ShardingEnv(
                        world_size=2,
                        rank=0,
                        pg=0,
                    ),
                    torch.device("cpu")
                )
            },
            device=torch.device("cpu"),
        )

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sqec_input_dist(features)
    """

    def __init__(
        self,
        input_feature_names: List[str],
        sharding_type_device_group_to_sharding: Dict[
            Tuple[str, Union[str, Tuple[str, ...]]],
            EmbeddingSharding[
                InferSequenceShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                List[torch.Tensor],
            ],
        ],
        device: Optional[torch.device] = None,
        feature_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._sharding_type_device_group_to_sharding = (
            sharding_type_device_group_to_sharding
        )
        self._input_dists = torch.nn.ModuleList([])
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        feature_names: List[str] = []
        for sharding in sharding_type_device_group_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist(device=device))
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))
        for f in feature_names:
            self._features_order.append(input_feature_names.index(f))

        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(
                self._features_order, device=feature_device, dtype=torch.int32
            ),
            persistent=False,
        )

    def forward(self, features: KeyedJaggedTensor) -> Tuple[
        List[KJTList],
        List[KeyedJaggedTensor],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
    ]:

        with torch.no_grad():
            ret: List[KJTList] = []
            unbucketize_permute_tensor = []
            bucket_mapping_tensor = []
            bucketized_lengths = []
            if self._features_order:
                features = input_dist_permute(
                    features,
                    self._features_order,
                    self._features_order_tensor,
                )
            features_by_sharding = (
                [features]
                if len(self._feature_splits) == 1
                else features.split(self._feature_splits)
            )

            for i in range(len(self._input_dists)):
                input_dist = self._input_dists[i]
                input_dist_result = input_dist(features_by_sharding[i])

                ret.append(input_dist_result.features)

                unbucketize_permute_tensor.append(
                    input_dist_result.unbucketize_permute_tensor
                )
                bucket_mapping_tensor.append(input_dist_result.bucket_mapping_tensor)
                bucketized_lengths.append(input_dist_result.bucketized_length)

            return (
                ret,
                features_by_sharding,
                unbucketize_permute_tensor,
                bucket_mapping_tensor,
                bucketized_lengths,
            )


class ShardedMCECLookup(torch.nn.Module):
    """
    This module implements distributed compute of a ShardedQuantManagedCollisionEmbeddingCollection.

    Args:
        managed_collision_collection (ShardedQuantManagedCollisionCollection): managed collision collection
        lookups (List[nn.Module]): embedding lookups

    Example::

    """

    def __init__(
        self,
        sharding: int,
        rank: int,
        mcc_remapper: ShardedMCCRemapper,
        ec_lookup: nn.Module,
    ) -> None:
        super().__init__()
        self._sharding = sharding
        self._rank = rank
        self._mcc_remapper = mcc_remapper
        self._ec_lookup = ec_lookup

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:
        remapped_kjt = self._mcc_remapper(features)
        return self._ec_lookup(remapped_kjt)


class ShardedQuantManagedCollisionEmbeddingCollection(ShardedQuantEmbeddingCollection):
    def __init__(
        self,
        module: QuantManagedCollisionEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        mc_sharder: InferManagedCollisionCollectionSharder,
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            module, table_name_to_parameter_sharding, env, fused_params, device
        )

        self._device = device
        self._env = env

        # TODO: This is a hack since _embedding_module doesn't need input
        # dist, so eliminating it so all fused a2a will ignore it.
        # we're using ec input_dist directly, so this cannot be escaped.
        # self._has_uninitialized_input_dist = False
        embedding_shardings = list(
            self._sharding_type_device_group_to_sharding.values()
        )

        self._managed_collision_collection: ShardedQuantManagedCollisionCollection = (
            mc_sharder.shard(
                module._managed_collision_collection,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
                # pyre-ignore
                embedding_shardings=embedding_shardings,
            )
        )
        self._return_remapped_features: bool = module._return_remapped_features
        self._create_mcec_lookups()

    def _create_mcec_lookups(self) -> None:
        mcec_lookups: List[nn.ModuleList] = []
        mcc_remappers: List[List[ShardedMCCRemapper]] = (
            self._managed_collision_collection.create_mcc_remappers()
        )
        for sharding in range(
            len(self._managed_collision_collection._embedding_shardings)
        ):
            ec_sharding_lookups = self._lookups[sharding]
            sharding_mcec_lookups: List[ShardedMCECLookup] = []
            for j, ec_lookup in enumerate(
                ec_sharding_lookups._embedding_lookups_per_rank  # pyre-ignore
            ):
                sharding_mcec_lookups.append(
                    ShardedMCECLookup(
                        sharding,
                        j,
                        mcc_remappers[sharding][j],
                        ec_lookup,
                    )
                )
            mcec_lookups.append(nn.ModuleList(sharding_mcec_lookups))
        self._mcec_lookup: nn.ModuleList = nn.ModuleList(mcec_lookups)

    # For consistency with ShardedManagedCollisionEmbeddingCollection
    @property
    def _embedding_collection(self) -> ShardedQuantEmbeddingCollection:
        return cast(ShardedQuantEmbeddingCollection, self)

    def input_dist(
        self,
        ctx: EmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> ListOfKJTList:
        # TODO: resolve incompatiblity with different contexts
        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False

        return self._managed_collision_collection.input_dist(
            # pyre-fixme [6]
            ctx,
            features,
        )

    def compute(
        self,
        ctx: ShrdCtx,
        dist_input: ListOfKJTList,
    ) -> List[List[torch.Tensor]]:
        ret: List[List[torch.Tensor]] = []
        for i in range(len(self._managed_collision_collection._embedding_shardings)):
            dist_input_i = dist_input[i]
            lookups = self._mcec_lookup[i]
            sharding_ret: List[torch.Tensor] = []
            for j, lookup in enumerate(lookups):
                rank_ret = lookup(
                    features=dist_input_i[j],
                )
                sharding_ret.append(rank_ret)
            ret.append(sharding_ret)
        return ret

    # pyre-ignore
    def output_dist(
        self,
        ctx: ShrdCtx,
        output: List[List[torch.Tensor]],
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:

        # pyre-ignore [6]
        ebc_out = super().output_dist(ctx, output)

        kjt_out: Optional[KeyedJaggedTensor] = None

        return ebc_out, kjt_out

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for fqn, _ in self.named_parameters():
            yield append_prefix(prefix, fqn)
        for fqn, _ in self.named_buffers():
            yield append_prefix(prefix, fqn)


class QuantManagedCollisionEmbeddingCollectionSharder(
    BaseQuantEmbeddingSharder[QuantManagedCollisionEmbeddingCollection]
):
    """
    This implementation uses non-fused EmbeddingCollection
    """

    def __init__(
        self,
        e_sharder: QuantEmbeddingCollectionSharder,
        mc_sharder: InferManagedCollisionCollectionSharder,
    ) -> None:
        super().__init__()
        self._e_sharder: QuantEmbeddingCollectionSharder = e_sharder
        self._mc_sharder: InferManagedCollisionCollectionSharder = mc_sharder

    def shardable_parameters(
        self, module: QuantManagedCollisionEmbeddingCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._e_sharder.shardable_parameters(module)

    def compute_kernels(
        self,
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.QUANT.value,
        ]

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return list(
            set.intersection(
                set(self._e_sharder.sharding_types(compute_device_type)),
                set(self._mc_sharder.sharding_types(compute_device_type)),
            )
        )

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        # TODO: to be deprecate after planner get cache_load_factor from ParameterConstraints
        return self._e_sharder.fused_params

    def shard(
        self,
        module: QuantManagedCollisionEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantManagedCollisionEmbeddingCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS not in fused_params:
            fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
                module,
                MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
                False,
            )
        if FUSED_PARAM_REGISTER_TBE_BOOL not in fused_params:
            fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
                module, FUSED_PARAM_REGISTER_TBE_BOOL, False
            )
        return ShardedQuantManagedCollisionEmbeddingCollection(
            module,
            params,
            self._mc_sharder,
            env,
            fused_params,
            device,
        )

    @property
    def module_type(self) -> Type[QuantManagedCollisionEmbeddingCollection]:
        return QuantManagedCollisionEmbeddingCollection
