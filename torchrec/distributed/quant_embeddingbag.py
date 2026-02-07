#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    FeatureShardingMixIn,
    GroupedEmbeddingConfig,
    InputDistOutputs,
    KJTList,
    ListOfKJTList,
)
from torchrec.distributed.embeddingbag import (
    construct_output_kt,
    create_sharding_infos_by_sharding_device_group,
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
from torchrec.distributed.sharding.cw_sharding import InferCwPooledEmbeddingSharding
from torchrec.distributed.sharding.rw_sharding import InferRwPooledEmbeddingSharding
from torchrec.distributed.sharding.tw_sharding import InferTwEmbeddingSharding
from torchrec.distributed.types import (
    NullShardedModuleContext,
    NullShardingContext,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import append_prefix, copy_to_device
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.pt2.checks import is_torchdynamo_compiling
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    FeatureProcessedEmbeddingBagCollection as QuantFeatureProcessedEmbeddingBagCollection,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    QuantManagedCollisionEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


def get_device_from_parameter_sharding(
    ps: ParameterSharding,
) -> Union[str, Tuple[str, ...]]:
    """
    Returns list of device type per shard if table is sharded across
    different device type, else reutrns single device type for the
    table parameter.
    """
    if not isinstance(ps.sharding_spec, EnumerableShardingSpec):
        raise ValueError("Expected EnumerableShardingSpec as input to the function")

    device_type_list: Tuple[str, ...] = tuple(
        [shard.placement.device().type for shard in ps.sharding_spec.shards]
    )
    if len(set(device_type_list)) == 1:
        return device_type_list[0]
    else:
        assert (
            ps.sharding_type == "row_wise"
        ), "Only row_wise sharding supports sharding across multiple device types for a table"
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


torch.fx.wrap("len")


@torch.fx.wrap
def flatten_feature_lengths(features: KeyedJaggedTensor) -> KeyedJaggedTensor:
    return features.flatten_lengths() if features.lengths().dim() > 1 else features


def create_infer_embedding_bag_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[
    NullShardingContext, InputDistOutputs, List[torch.Tensor], torch.Tensor
]:
    propogate_device: bool = get_propogate_device()
    storage_device_type_from_sharding_infos: Union[str, Tuple[str, ...]] = (
        get_device_from_sharding_infos(sharding_infos)
    )
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwEmbeddingSharding(
            sharding_infos, env, device=device if propogate_device else None
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return InferRwPooledEmbeddingSharding(
            sharding_infos,
            env,
            device=device if propogate_device else None,
            device_type_from_sharding_infos=storage_device_type_from_sharding_infos,
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return InferCwPooledEmbeddingSharding(
            sharding_infos,
            env,
            device=device if propogate_device else None,
            permute_embeddings=True,
        )
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


class ShardedQuantEmbeddingBagCollection(
    ShardedQuantEmbeddingModuleState[
        ListOfKJTList,
        List[List[torch.Tensor]],
        KeyedTensor,
        NullShardedModuleContext,
    ],
):
    """
    Sharded implementation of `EmbeddingBagCollection`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],  # support for Hybrid Sharding
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._embedding_bag_configs: List[EmbeddingBagConfig] = (
            module.embedding_bag_configs()
        )
        self._sharding_type_device_group_to_sharding_infos: Dict[
            Tuple[str, Union[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]
        ] = create_sharding_infos_by_sharding_device_group(
            module, table_name_to_parameter_sharding, "embedding_bags.", fused_params
        )
        self._sharding_type_device_group_to_sharding: Dict[
            Tuple[str, Union[str, Tuple[str, ...]]],
            EmbeddingSharding[
                NullShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ] = {
            (sharding_type, device_group): create_infer_embedding_bag_sharding(
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
        self._device = device
        self._is_weighted: bool = module.is_weighted()
        self._lookups: List[nn.Module] = []
        self._create_lookups(fused_params, device)
        self._fused_params = fused_params

        # Ensure output dist is set for post processing from an inference runtime (ie. setting device from runtime).
        self._output_dists: torch.nn.ModuleList = torch.nn.ModuleList()

        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []

        # forward pass flow control
        self._has_uninitialized_output_dist: bool = True

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
                tables_weights_prefix="embedding_bags",
            )
        else:
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

            self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
            for table in self._embedding_bag_configs:
                self.embedding_bags[table.name] = torch.nn.Module()

            for _, lookup in zip(
                self._sharding_type_device_group_to_sharding.keys(), self._lookups
            ):
                lookup_state_dict = lookup.state_dict()
                for key in lookup_state_dict:
                    if key.endswith(".weight"):
                        table_name = key[: -len(".weight")]
                        self.embedding_bags[table_name].register_buffer(
                            "weight", lookup_state_dict[key]
                        )

        self._input_dist_module: ShardedQuantEbcInputDist = ShardedQuantEbcInputDist(
            self._sharding_type_device_group_to_sharding, self._device
        )

    def tbes_configs(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return self._tbes_configs

    def sharding_type_device_group_to_sharding_infos(
        self,
    ) -> Dict[Tuple[str, Union[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]]:
        return self._sharding_type_device_group_to_sharding_infos

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._lookups.append(
                sharding.create_lookup(
                    device=device,
                    fused_params=fused_params,
                )
            )

    def _create_output_dist(self, device: Optional[torch.device] = None) -> None:
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    def input_dist(
        self, ctx: NullShardedModuleContext, features: KeyedJaggedTensor
    ) -> ListOfKJTList:
        input_dist_outputs = self._input_dist_module(features)

        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False

        return input_dist_outputs

    def compute(
        self,
        ctx: NullShardedModuleContext,
        dist_input: ListOfKJTList,
    ) -> List[List[torch.Tensor]]:
        # syntax for torchscript
        return [lookup.forward(dist_input[i]) for i, lookup in enumerate(self._lookups)]

    def output_dist(
        self,
        ctx: NullShardedModuleContext,
        output: List[List[torch.Tensor]],
    ) -> KeyedTensor:
        return construct_output_kt(
            embeddings=[
                dist.forward(output[i]) for i, dist in enumerate(self._output_dists)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: NullShardedModuleContext, input: ListOfKJTList
    ) -> KeyedTensor:
        return self.output_dist(ctx, self.compute(ctx, input))

    def forward(self, *input, **kwargs) -> KeyedTensor:
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input, **kwargs)
        return self.compute_and_output_dist(ctx, dist_input)

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return super().copy(device)

    @property
    def shardings(
        self,
    ) -> Dict[Tuple[str, Union[str, Tuple[str, ...]]], FeatureShardingMixIn]:
        return self._sharding_type_device_group_to_sharding

    def create_context(self) -> NullShardedModuleContext:
        if is_torchdynamo_compiling():
            # Context creation is not supported by dynamo yet.
            # Context is not needed for TW sharding =>
            # Unblocking dynamo TW with None.
            return None

        return NullShardedModuleContext()


class QuantEmbeddingBagCollectionSharder(
    BaseQuantEmbeddingSharder[QuantEmbeddingBagCollection]
):
    def shard(
        self,
        module: QuantEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
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

        return ShardedQuantEmbeddingBagCollection(
            module, params, env, fused_params, device=device
        )

    @property
    def module_type(self) -> Type[QuantEmbeddingBagCollection]:
        return QuantEmbeddingBagCollection


class ShardedQuantFeatureProcessedEmbeddingBagCollection(
    ShardedQuantEmbeddingBagCollection,
):
    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],  # support for hybrid sharding
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        feature_processor: Optional[FeatureProcessorsCollection] = None,
    ) -> None:
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            env,
            fused_params,
            device,
        )
        assert feature_processor is not None
        device_type: str = self._device.type if self._device is not None else "cuda"
        self.feature_processors_per_rank: nn.ModuleList = torch.nn.ModuleList()
        feature_processor_device = None
        for _, param in feature_processor.named_parameters():
            if feature_processor_device is None:
                feature_processor_device = param.device
            elif feature_processor_device != param.device:
                raise RuntimeError(
                    f"Feature processor has inconsistent devices. Expected {feature_processor_device}, got {param.device}"
                )

        for _, buffer in feature_processor.named_buffers():
            if feature_processor_device is None:
                feature_processor_device = buffer.device
            elif feature_processor_device != buffer.device:
                raise RuntimeError(
                    f"Feature processor has inconsistent devices. Expected {feature_processor_device}, got {param.device}"
                )

        if isinstance(env, Dict):
            expected_device_type = "cuda"
            for (
                embedding_configs
            ) in self._sharding_type_device_group_to_sharding_infos.values():
                # throws if not all shards only have the expected device type
                shard_device_type = get_device_from_sharding_infos(embedding_configs)
                if isinstance(shard_device_type, tuple):
                    raise RuntimeError(
                        f"Sharding across multiple device types for FeatureProcessedEmbeddingBagCollection is not supported yet, got {shard_device_type}",
                    )
                assert (
                    shard_device_type == expected_device_type
                ), f"Expected {expected_device_type} but got {shard_device_type} for FeatureProcessedEmbeddingBagCollection sharding device type"

            # TODO(hcxu): support hybrid sharding with feature_processors_per_rank: ModuleList(ModuleList()), if compatible
            world_size = env[expected_device_type].world_size
        else:
            world_size = env.world_size

        if feature_processor_device is None:
            self.feature_processors_per_rank += [feature_processor] * world_size
        else:
            for i in range(world_size):
                # Generic copy, for example initailized on cpu but -> sharding as meta
                self.feature_processors_per_rank.append(
                    copy.deepcopy(feature_processor)
                    if device_type == "meta"
                    else copy_to_device(
                        feature_processor,
                        feature_processor_device,
                        (
                            torch.device(f"{device_type}:{i}")
                            if device_type == "cuda"
                            else torch.device(f"{device_type}")
                        ),
                    )
                )

    def apply_feature_processor(
        self,
        kjt_list: KJTList,
    ) -> KJTList:
        l: List[KeyedJaggedTensor] = []
        for i in range(len(self.feature_processors_per_rank)):
            l.append(self.feature_processors_per_rank[i](kjt_list[i]))
        return KJTList(l)

    def compute(
        self,
        ctx: NullShardedModuleContext,
        dist_input: ListOfKJTList,  # List_per_sharding[List_per_rank[KJT]]
    ) -> List[List[torch.Tensor]]:
        return [
            lookup.forward(self.apply_feature_processor(dist_input[i]))
            for i, lookup in enumerate(self._lookups)
        ]


class QuantFeatureProcessedEmbeddingBagCollectionSharder(
    BaseQuantEmbeddingSharder[QuantFeatureProcessedEmbeddingBagCollection]
):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.QUANT.value]

    def shard(
        self,
        module: QuantFeatureProcessedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        qebc = module
        assert isinstance(qebc, QuantEmbeddingBagCollection)
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(qebc.output_dtype())
        )
        if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS not in fused_params:
            fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
                qebc, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            )
        if FUSED_PARAM_REGISTER_TBE_BOOL not in fused_params:
            fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
                qebc, FUSED_PARAM_REGISTER_TBE_BOOL, False
            )

        return ShardedQuantFeatureProcessedEmbeddingBagCollection(
            qebc,
            params,
            env,
            fused_params,
            device=device,
            feature_processor=module.feature_processor,
        )

    @property
    def module_type(self) -> Type[QuantFeatureProcessedEmbeddingBagCollection]:
        return QuantFeatureProcessedEmbeddingBagCollection


class ShardedQuantEbcInputDist(torch.nn.Module):
    """
    This module implements distributed inputs of a ShardedQuantEmbeddingBagCollection.

    Args:
        sharding_type_device_group_to_sharding (Dict[
            Tuple[str, str],
            EmbeddingSharding[
                NullShardingContext,
                KJTList,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ]): map from sharding type to EmbeddingSharding.
        device (Optional[torch.device]): default compute device.

    Example::

        sqebc_input_dist = ShardedQuantEbcInputDist(
            sharding_type_device_group_to_sharding={
                (ShardingType.TABLE_WISE, "cpu"): InferTwSequenceEmbeddingSharding(
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

        sqebc_input_dist(features)
    """

    def __init__(
        self,
        sharding_type_device_group_to_sharding: Dict[
            Tuple[str, Union[str, Tuple[str, ...]]],
            EmbeddingSharding[
                NullShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._sharding_type_device_group_to_sharding = (
            sharding_type_device_group_to_sharding
        )
        self._device = device

        self._shardings: List[
            EmbeddingSharding[
                NullShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                torch.Tensor,
            ]
        ] = list(sharding_type_device_group_to_sharding.values())

        self._input_dists: List[nn.Module] = []

        self._features_order: List[int] = []
        self._feature_names: List[List[str]] = [
            sharding.feature_names() for sharding in self._shardings
        ]
        self._feature_splits: List[int] = [
            len(sharding) for sharding in self._feature_names
        ]

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_features_permute: bool = True

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        features_device: torch.device,
        input_dist_device: Optional[torch.device] = None,
    ) -> None:
        flat_feature_names: List[str] = [
            feature_name
            for sharding_feature_name in self._feature_names
            for feature_name in sharding_feature_name
        ]
        self._input_dists = [
            sharding.create_input_dist(device=input_dist_device)
            for sharding in self._shardings
        ]

        if flat_feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in flat_feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=features_device, dtype=torch.int32
                ),
                persistent=False,
            )

    def forward(self, features: KeyedJaggedTensor) -> ListOfKJTList:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            ListOfKJTList
        """
        if self._has_uninitialized_input_dist:
            self._create_input_dist(
                features.keys(),
                features.device(),
                self._device,
            )
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    #  but got `Union[Module, Tensor]`.
                    self._features_order_tensor,
                )
            else:
                features = flatten_feature_lengths(features)
            features_by_shards = (
                [features]
                if len(self._feature_splits) == 1
                else features.split(self._feature_splits)
            )
            return ListOfKJTList(
                [
                    self._input_dists[i].forward(features_by_shards[i]).features
                    for i in range(len(self._input_dists))
                ]
            )


class ShardedMCEBCLookup(torch.nn.Module):
    """
    This module implements distributed compute of a ShardedQuantManagedCollisionEmbeddingBagCollection.

    Args:
        sharding (int): sharding index
        rank (int): rank index
        mcc_remapper (ShardedMCCRemapper): managed collision collection remapper
        ebc_lookup (nn.Module): embedding bag collection lookup

    Example::

    """

    def __init__(
        self,
        sharding: int,
        rank: int,
        mcc_remapper: ShardedMCCRemapper,
        ebc_lookup: nn.Module,
    ) -> None:
        super().__init__()
        self._sharding = sharding
        self._rank = rank
        self._mcc_remapper = mcc_remapper
        self._ebc_lookup = ebc_lookup

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:
        """
        Applies managed collision collection remapping and performs embedding lookup.

        Args:
            features (KeyedJaggedTensor): input features

        Returns:
            Tuple[Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]]: embedding lookup result
        """
        remapped_kjt = self._mcc_remapper(features)
        return self._ebc_lookup(remapped_kjt)


class ShardedQuantManagedCollisionEmbeddingBagCollection(
    ShardedQuantEmbeddingBagCollection
):
    def __init__(
        self,
        module: QuantManagedCollisionEmbeddingBagCollection,
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
                embedding_shardings=embedding_shardings,
            )
        )
        self._return_remapped_features: bool = module._return_remapped_features
        self._create_mcebc_lookups()

    def _create_mcebc_lookups(self) -> None:
        mcebc_lookups: List[nn.ModuleList] = []
        mcc_remappers: List[List[ShardedMCCRemapper]] = (
            self._managed_collision_collection.create_mcc_remappers()
        )
        for sharding in range(
            len(self._managed_collision_collection._embedding_shardings)
        ):
            ebc_sharding_lookups = self._lookups[sharding]
            sharding_mcebc_lookups: List[ShardedMCEBCLookup] = []
            for j, ec_lookup in enumerate(
                ebc_sharding_lookups._embedding_lookups_per_rank
            ):
                sharding_mcebc_lookups.append(
                    ShardedMCEBCLookup(
                        sharding,
                        j,
                        mcc_remappers[sharding][j],
                        ec_lookup,
                    )
                )
            mcebc_lookups.append(nn.ModuleList(sharding_mcebc_lookups))
        self._mcebc_lookup: nn.ModuleList = nn.ModuleList(mcebc_lookups)

    def input_dist(
        self,
        ctx: NullShardedModuleContext,
        features: KeyedJaggedTensor,
    ) -> ListOfKJTList:
        # TODO: resolve incompatiblity with different contexts
        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False

        return self._managed_collision_collection.input_dist(
            ctx,
            features,
            is_sequence_embedding=False,
        )

    def compute(
        self,
        ctx: NullShardedModuleContext,
        dist_input: ListOfKJTList,
    ) -> List[List[torch.Tensor]]:
        ret: List[List[torch.Tensor]] = []
        for i in range(len(self._managed_collision_collection._embedding_shardings)):
            dist_input_i = dist_input[i]
            lookups = self._mcebc_lookup[i]
            sharding_ret: List[torch.Tensor] = []
            for j, lookup in enumerate(lookups):
                rank_ret = lookup(
                    features=dist_input_i[j],
                )
                sharding_ret.append(rank_ret)
            ret.append(sharding_ret)
        return ret

    def output_dist(
        self,
        ctx: NullShardedModuleContext,
        output: List[List[torch.Tensor]],
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:

        ebc_out = super().output_dist(ctx, output)

        kjt_out: Optional[KeyedJaggedTensor] = None

        return ebc_out, kjt_out

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for fqn, _ in self.named_parameters():
            yield append_prefix(prefix, fqn)
        for fqn, _ in self.named_buffers():
            yield append_prefix(prefix, fqn)


class QuantManagedCollisionEmbeddingBagCollectionSharder(
    BaseQuantEmbeddingSharder[QuantManagedCollisionEmbeddingBagCollection]
):
    """
    Sharder for QuantManagedCollisionEmbeddingBagCollection.

    This implementation uses non-fused EmbeddingBagCollection and manages both
    embedding bag collection sharding and managed collision collection sharding.

    Args:
        e_sharder (QuantEmbeddingBagCollectionSharder): sharder for embedding bag collection
        mc_sharder (InferManagedCollisionCollectionSharder): sharder for managed collision collection

    Example::

    """

    def __init__(
        self,
        e_sharder: QuantEmbeddingBagCollectionSharder,
        mc_sharder: InferManagedCollisionCollectionSharder,
    ) -> None:
        super().__init__()
        self._e_sharder: QuantEmbeddingBagCollectionSharder = e_sharder
        self._mc_sharder: InferManagedCollisionCollectionSharder = mc_sharder

    def shardable_parameters(
        self, module: QuantManagedCollisionEmbeddingBagCollection
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
        module: QuantManagedCollisionEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantManagedCollisionEmbeddingBagCollection:
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
        return ShardedQuantManagedCollisionEmbeddingBagCollection(
            module,
            params,
            self._mc_sharder,
            env,
            fused_params,
            device,
        )

    @property
    def module_type(self) -> Type[QuantManagedCollisionEmbeddingBagCollection]:
        return QuantManagedCollisionEmbeddingBagCollection
