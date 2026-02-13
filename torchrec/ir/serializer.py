#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from torchrec.ir.schema import (
    EBCMetadata,
    EmbeddingBagConfigMetadata,
    FPEBCMetadata,
    KTRegroupAsDictMetadata,
    PositionWeightedModuleCollectionMetadata,
    PositionWeightedModuleMetadata,
)
from torchrec.ir.types import SerializerInterface
from torchrec.ir.utils import logging, qualname
from torchrec.modules.embedding_configs import (
    data_type_to_dtype,
    DataType,
    EmbeddingBagConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    FeatureProcessor,
    FeatureProcessorsCollection,
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.regroup import KTRegroupAsDict
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


logger: logging.Logger = logging.getLogger(__name__)


def embedding_bag_config_to_metadata(
    table_config: EmbeddingBagConfig,
) -> EmbeddingBagConfigMetadata:
    return EmbeddingBagConfigMetadata(
        num_embeddings=table_config.num_embeddings,
        embedding_dim=table_config.embedding_dim,
        name=table_config.name,
        # pyrefly: ignore[bad-argument-type]
        data_type=table_config.data_type.value,
        feature_names=table_config.feature_names,
        weight_init_max=table_config.weight_init_max,
        weight_init_min=table_config.weight_init_min,
        need_pos=table_config.need_pos,
        # pyrefly: ignore[bad-argument-type]
        pooling=table_config.pooling.value,
    )


def embedding_metadata_to_config(
    table_config: EmbeddingBagConfigMetadata,
) -> EmbeddingBagConfig:
    return EmbeddingBagConfig(
        num_embeddings=table_config.num_embeddings,
        embedding_dim=table_config.embedding_dim,
        name=table_config.name,
        data_type=DataType(table_config.data_type),
        feature_names=table_config.feature_names,
        weight_init_max=table_config.weight_init_max,
        weight_init_min=table_config.weight_init_min,
        need_pos=table_config.need_pos,
        pooling=PoolingType(table_config.pooling),
    )


def get_deserialized_device(
    config_device: Optional[str], device: Optional[torch.device]
) -> Optional[torch.device]:
    if config_device:
        original_device = torch.device(config_device)
        if device is None:
            device = original_device
        elif original_device.type != device.type:
            logger.warning(
                f"deserialized device={device} overrides the original device={original_device}"
            )
    return device


def ebc_meta_forward(
    ebc: EmbeddingBagCollection,
    features: KeyedJaggedTensor,
) -> KeyedTensor:
    batch_size = features.stride()
    dims = ebc._lengths_per_embedding
    arg_list = [
        features.values(),
        features.weights_or_none(),
        features.lengths_or_none(),
        features.offsets_or_none(),
    ]  # if want to include the weights: `+ [bag.weight for bag in self.embedding_bags.values()]`
    outputs = torch.ops.torchrec.ir_emb_lookup(arg_list, batch_size, dims)
    return KeyedTensor(
        keys=ebc._embedding_names,
        values=torch.cat(outputs, dim=1),
        length_per_key=ebc._lengths_per_embedding,
    )


def fpebc_meta_forward(
    fpebc: FeatureProcessedEmbeddingBagCollection,
    features: KeyedJaggedTensor,
) -> KeyedTensor:
    batch_size = features.stride()
    ebc = fpebc._embedding_bag_collection
    dims = ebc._lengths_per_embedding
    arg_list = [
        features.values(),
        features.weights_or_none(),
        features.lengths_or_none(),
        features.offsets_or_none(),
    ]  # if want to include the weights: `+ [bag.weight for bag in self.embedding_bags.values()]`
    outputs = torch.ops.torchrec.ir_emb_lookup(arg_list, batch_size, dims)
    return KeyedTensor(
        keys=ebc._embedding_names,
        values=torch.cat(outputs, dim=1),
        length_per_key=ebc._lengths_per_embedding,
    )


def kt_regroup_meta_forward(
    op_module: KTRegroupAsDict, keyed_tensors: List[KeyedTensor]
) -> Dict[str, torch.Tensor]:
    lengths_dict: Dict[str, int] = {}
    batch_size = keyed_tensors[0].values().size(0)
    for kt in keyed_tensors:
        for key, length in zip(kt.keys(), kt.length_per_key()):
            lengths_dict[key] = length
    out_lengths: List[int] = [0] * len(op_module._groups)
    for i, group in enumerate(op_module._groups):
        out_lengths[i] = sum(lengths_dict[key] for key in group)
    arg_list = [kt.values() for kt in keyed_tensors]
    dtype = (
        data_type_to_dtype(op_module._emb_dtype)
        if op_module._emb_dtype
        else torch.float32
    )
    outputs = torch.ops.torchrec.ir_kt_regroup(
        arg_list,
        batch_size,
        out_lengths,
        dtype=dtype,
    )
    return dict(zip(op_module._keys, outputs))


class JsonSerializer(SerializerInterface):
    """
    Serializer for torch.export IR using json.
    """

    module_to_serializer_cls: Dict[str, Type["JsonSerializer"]] = {}
    _module_cls: Optional[Type[nn.Module]] = None
    _children: Optional[List[str]] = None

    @classmethod
    def children(cls, module: nn.Module) -> List[str]:
        return [] if not cls._children else cls._children

    @classmethod
    def serialize_to_dict(cls, module: nn.Module) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        raise NotImplementedError()

    @classmethod
    def swap_meta_forward(cls, module: nn.Module) -> None:
        pass

    @classmethod
    def encapsulate_module(cls, module: nn.Module) -> List[str]:
        typename = qualname(module)
        serializer = cls.module_to_serializer_cls.get(typename)
        if serializer is None:
            raise ValueError(
                f"Expected typename to be one of {list(cls.module_to_serializer_cls.keys())}, got {typename}"
            )
        assert issubclass(serializer, JsonSerializer)
        assert serializer._module_cls is not None
        if not isinstance(module, serializer._module_cls):
            raise ValueError(
                f"Expected module to be of type {qualname(serializer._module_cls)}, "
                f"got {qualname(module)}"
            )
        metadata_dict = serializer.serialize_to_dict(module)
        raw_dict = {"typename": typename, "metadata_dict": metadata_dict}
        ir_metadata_tensor = torch.frombuffer(
            json.dumps(raw_dict).encode(), dtype=torch.uint8
        )
        module.register_buffer("ir_metadata", ir_metadata_tensor, persistent=False)
        serializer.swap_meta_forward(module)
        return serializer.children(module)

    @classmethod
    def decapsulate_module(
        cls, module: nn.Module, device: Optional[torch.device] = None
    ) -> nn.Module:
        raw_bytes = module.get_buffer("ir_metadata").numpy().tobytes()
        raw_dict = json.loads(raw_bytes.decode())
        typename = raw_dict["typename"]
        metadata_dict = raw_dict["metadata_dict"]
        if typename not in cls.module_to_serializer_cls:
            raise ValueError(
                f"Expected typename to be one of {list(cls.module_to_serializer_cls.keys())}, got {typename}"
            )
        serializer = cls.module_to_serializer_cls[typename]
        assert issubclass(serializer, JsonSerializer)
        module = serializer.deserialize_from_dict(metadata_dict, device, module)

        if serializer._module_cls is None:
            raise ValueError(
                "Must assign a nn.Module to class static variable _module_cls"
            )
        if not isinstance(module, serializer._module_cls):
            raise ValueError(
                f"Expected module to be of type {qualname(serializer._module_cls)}, got {qualname(module)}"
            )
        return module


class EBCJsonSerializer(JsonSerializer):
    _module_cls = EmbeddingBagCollection

    @classmethod
    def swap_meta_forward(cls, module: nn.Module) -> None:
        # pyrefly: ignore[bad-argument-type]
        assert isinstance(module, cls._module_cls)
        module.forward = ebc_meta_forward.__get__(module, cls._module_cls)

    @classmethod
    def serialize_to_dict(
        cls,
        module: nn.Module,
    ) -> Dict[str, Any]:
        ebc_metadata = EBCMetadata(
            tables=[
                embedding_bag_config_to_metadata(table_config)
                # pyrefly: ignore[not-callable]
                for table_config in module.embedding_bag_configs()
            ],
            # pyrefly: ignore[not-callable]
            is_weighted=module.is_weighted(),
            device=str(module.device),
        )
        ebc_metadata_dict = ebc_metadata.__dict__
        ebc_metadata_dict["tables"] = [
            table_config.__dict__ for table_config in ebc_metadata_dict["tables"]
        ]
        return ebc_metadata_dict

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        tables = [
            EmbeddingBagConfigMetadata(**table_config)
            for table_config in metadata_dict["tables"]
        ]

        device = get_deserialized_device(metadata_dict.get("device"), device)
        return EmbeddingBagCollection(
            tables=[
                embedding_metadata_to_config(table_config) for table_config in tables
            ],
            is_weighted=metadata_dict["is_weighted"],
            device=device,
        )


JsonSerializer.module_to_serializer_cls[qualname(EmbeddingBagCollection)] = (
    EBCJsonSerializer
)


class PWMJsonSerializer(JsonSerializer):
    _module_cls = PositionWeightedModule

    @classmethod
    def serialize_to_dict(cls, module: nn.Module) -> Dict[str, Any]:
        metadata = PositionWeightedModuleMetadata(
            # pyrefly: ignore[bad-argument-type, bad-index]
            max_feature_length=module.position_weight.shape[0],
        )
        return metadata.__dict__

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        metadata = PositionWeightedModuleMetadata(**metadata_dict)
        return PositionWeightedModule(metadata.max_feature_length, device)


JsonSerializer.module_to_serializer_cls[qualname(PositionWeightedModule)] = (
    PWMJsonSerializer
)


class PWMCJsonSerializer(JsonSerializer):
    _module_cls = PositionWeightedModuleCollection

    @classmethod
    def serialize_to_dict(cls, module: nn.Module) -> Dict[str, Any]:
        metadata = PositionWeightedModuleCollectionMetadata(
            max_feature_lengths=[  # convert to list of tuples to preserve the order
                (feature, len)
                #  attribute `items`.
                # pyrefly: ignore[missing-attribute]
                for feature, len in module.max_feature_lengths.items()
            ],
        )
        return metadata.__dict__

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        metadata = PositionWeightedModuleCollectionMetadata(**metadata_dict)
        max_feature_lengths = {
            feature: len for feature, len in metadata.max_feature_lengths
        }
        return PositionWeightedModuleCollection(max_feature_lengths, device)


JsonSerializer.module_to_serializer_cls[qualname(PositionWeightedModuleCollection)] = (
    PWMCJsonSerializer
)


class FPEBCJsonSerializer(JsonSerializer):
    _module_cls = FeatureProcessedEmbeddingBagCollection
    _children = ["_feature_processors", "_embedding_bag_collection"]

    @classmethod
    def serialize_to_dict(
        cls,
        module: nn.Module,
    ) -> Dict[str, Any]:
        if isinstance(module._feature_processors, FeatureProcessorsCollection):
            metadata = FPEBCMetadata(
                is_fp_collection=True,
                features=[],
            )
        else:
            metadata = FPEBCMetadata(
                is_fp_collection=False,
                #  attribute `keys`.
                # pyrefly: ignore[missing-attribute]
                features=list(module._feature_processors.keys()),
            )
        return metadata.__dict__

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        metadata = FPEBCMetadata(**metadata_dict)
        assert unflatten_ep is not None
        if metadata.is_fp_collection:
            # pyrefly: ignore[bad-assignment]
            feature_processors = unflatten_ep._feature_processors
            assert isinstance(feature_processors, FeatureProcessorsCollection)
        else:
            feature_processors: dict[str, FeatureProcessor] = {}
            for feature in metadata.features:
                fp = getattr(unflatten_ep._feature_processors, feature)
                assert isinstance(fp, FeatureProcessor)
                feature_processors[feature] = fp
        ebc = unflatten_ep._embedding_bag_collection
        assert isinstance(ebc, EmbeddingBagCollection)
        return FeatureProcessedEmbeddingBagCollection(
            ebc,
            feature_processors,
        )


JsonSerializer.module_to_serializer_cls[
    qualname(FeatureProcessedEmbeddingBagCollection)
] = FPEBCJsonSerializer


class KTRegroupAsDictJsonSerializer(JsonSerializer):
    _module_cls = KTRegroupAsDict

    @classmethod
    def swap_meta_forward(cls, module: nn.Module) -> None:
        # pyrefly: ignore[bad-argument-type]
        assert isinstance(module, cls._module_cls)
        module.forward = kt_regroup_meta_forward.__get__(module, cls._module_cls)

    @classmethod
    def serialize_to_dict(
        cls,
        module: nn.Module,
    ) -> Dict[str, Any]:
        metadata = KTRegroupAsDictMetadata(
            #  `Union[Module, Tensor]`.
            # pyrefly: ignore[bad-argument-type]
            keys=module._keys,
            #  `Union[Module, Tensor]`.
            # pyrefly: ignore[bad-argument-type]
            groups=module._groups,
            emb_dtype=(
                # pyrefly: ignore[missing-attribute]
                module._emb_dtype.value
                if module._emb_dtype is not None
                else None
            ),
        )
        return metadata.__dict__

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        metadata = KTRegroupAsDictMetadata(**metadata_dict)
        return KTRegroupAsDict(
            keys=metadata.keys,
            groups=metadata.groups,
            emb_dtype=(
                DataType(metadata.emb_dtype) if metadata.emb_dtype is not None else None
            ),
        )


JsonSerializer.module_to_serializer_cls[qualname(KTRegroupAsDict)] = (
    KTRegroupAsDictJsonSerializer
)
