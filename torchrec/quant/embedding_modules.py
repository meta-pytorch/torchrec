#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import itertools
from collections import defaultdict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
)
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import KVEmbeddingInference
from torch import Tensor
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    DATA_TYPE_NUM_BITS,
    data_type_to_sparse_type,
    DataType,
    dtype_to_data_type,
    EmbeddingBagConfig,
    EmbeddingConfig,
    pooling_type_to_pooling_mode,
    PoolingType,
    QuantConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection as OriginalEmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection as OriginalEmbeddingCollection,
    EmbeddingCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.modules.fp_embedding_modules import (
    FeatureProcessedEmbeddingBagCollection as OriginalFeatureProcessedEmbeddingBagCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingCollection as OriginalManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.modules.utils import (
    _get_batching_hinted_output,
    construct_jagged_tensors_inference,
)
from torchrec.sparse.jagged_tensor import (
    ComputeKJTToJTDict,
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.tensor_types import UInt2Tensor, UInt4Tensor
from torchrec.types import ModuleNoCopyMixin

torch.fx.wrap("_get_batching_hinted_output")
torch.fx.wrap("len")

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except (OSError, RuntimeError):
    pass

# OSS
try:
    pass
except ImportError:
    pass

MODULE_ATTR_REGISTER_TBES_BOOL: str = "__register_tbes_in_named_modules"

MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: str = (
    "__quant_state_dict_split_scale_bias"
)

MODULE_ATTR_ROW_ALIGNMENT_INT: str = "__register_row_alignment_in_named_modules"

MODULE_ATTR_EMB_CONFIG_NAME_TO_NUM_ROWS_POST_PRUNING_DICT: str = (
    "__emb_name_to_num_rows_post_pruning"
)

MODULE_ATTR_REMOVE_STBE_PADDING_BOOL: str = "__remove_stbe_padding"

MODULE_ATTR_USE_UNFLATTENED_LENGTHS_FOR_BATCHING: str = (
    "__use_unflattened_lengths_for_batching"
)

MODULE_ATTR_USE_BATCHING_HINTED_OUTPUT: str = "__use_batching_hinted_output"

MODULE_ATTR_CACHE_FEATURES_ORDER: str = "__cache_features_order"

DEFAULT_ROW_ALIGNMENT = 16


@torch.fx.wrap
def _get_feature_length(feature: KeyedJaggedTensor) -> Tensor:
    return feature.lengths()


@torch.fx.wrap
def _get_kjt_keys(feature: KeyedJaggedTensor) -> List[str]:
    # this is a fx rule to help with batching hinting jagged sequence tensor coalescing.
    return feature.keys()


@torch.fx.wrap
def _permute_kjt(
    features: KeyedJaggedTensor,
    permute_order: List[int],
    permute_order_tensor: Optional[Tensor] = None,
) -> KeyedJaggedTensor:
    if permute_order == list(range(len(permute_order))):
        return features.flatten_lengths()
    return features.permute(permute_order, permute_order_tensor)


@torch.fx.wrap
def _cat_embeddings(embeddings: List[Tensor]) -> Tensor:
    return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=1)


@torch.fx.wrap
def _get_unflattened_lengths(lengths: torch.Tensor, num_features: int) -> torch.Tensor:
    """
    Unflatten lengths tensor from [F * B] to [F, B].
    """
    return lengths.view(num_features, -1)


def for_each_module_of_type_do(
    module: nn.Module,
    module_types: List[Type[torch.nn.Module]],
    op: Callable[[torch.nn.Module], None],
) -> None:
    for m in module.modules():
        if any([isinstance(m, t) for t in module_types]):
            op(m)


def quant_prep_enable_quant_state_dict_split_scale_bias(module: nn.Module) -> None:
    setattr(module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, True)


def quant_prep_enable_quant_state_dict_split_scale_bias_for_types(
    module: nn.Module, module_types: List[Type[torch.nn.Module]]
) -> None:
    for_each_module_of_type_do(
        module,
        module_types,
        lambda m: setattr(m, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, True),
    )


def quant_prep_enable_register_tbes(
    module: nn.Module, module_types: List[Type[torch.nn.Module]]
) -> None:
    for_each_module_of_type_do(
        module,
        module_types,
        lambda m: setattr(m, MODULE_ATTR_REGISTER_TBES_BOOL, True),
    )


def quant_prep_customize_row_alignment(
    module: nn.Module, module_types: List[Type[torch.nn.Module]], row_alignment: int
) -> None:
    for_each_module_of_type_do(
        module,
        module_types,
        lambda m: setattr(m, MODULE_ATTR_ROW_ALIGNMENT_INT, row_alignment),
    )


def quant_prep_enable_cache_features_order(
    module: nn.Module, module_types: List[Type[torch.nn.Module]]
) -> None:
    for_each_module_of_type_do(
        module,
        module_types,
        lambda m: setattr(m, MODULE_ATTR_CACHE_FEATURES_ORDER, True),
    )


def quantize_state_dict(
    module: nn.Module,
    table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]],
    table_name_to_data_type: Dict[str, DataType],
    table_name_to_num_embeddings_post_pruning: Optional[Dict[str, int]] = None,
) -> torch.device:
    device = torch.device("cpu")
    if not table_name_to_num_embeddings_post_pruning:
        table_name_to_num_embeddings_post_pruning = {}

    for key, tensor in module.state_dict().items():
        # Extract table name from state dict key.
        # e.g. ebc.embedding_bags.t1.weight
        splits = key.split(".")
        assert splits[-1] == "weight"
        table_name = splits[-2]
        data_type = table_name_to_data_type[table_name]
        num_rows = tensor.shape[0]

        if table_name in table_name_to_num_embeddings_post_pruning:
            num_rows = table_name_to_num_embeddings_post_pruning[table_name]

        device = tensor.device
        num_bits = DATA_TYPE_NUM_BITS[data_type]

        if tensor.is_meta:
            quant_weight = torch.empty(
                (num_rows, (tensor.shape[1] * num_bits) // 8),
                device="meta",
                dtype=torch.uint8,
            )
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                scale_shift = torch.empty(
                    (num_rows, 4),
                    device="meta",
                    dtype=torch.uint8,
                )
            else:
                scale_shift = None
        else:
            if num_rows != tensor.shape[0]:
                tensor = tensor[:num_rows, :]
            if tensor.dtype == torch.float or tensor.dtype == torch.float16:
                if data_type == DataType.FP16:
                    if tensor.dtype == torch.float:
                        tensor = tensor.half()
                    quant_res = tensor.view(torch.uint8)
                elif data_type == DataType.FP32:
                    if tensor.dtype == torch.float16:
                        tensor = tensor.float()
                    quant_res = tensor.view(torch.uint8)
                else:
                    quant_res = (
                        torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                            tensor, num_bits
                        )
                    )
            else:
                raise Exception("Unsupported dtype: {tensor.dtype}")
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                quant_weight, scale_shift = (
                    quant_res[:, :-4],
                    quant_res[:, -4:],
                )
            else:
                quant_weight, scale_shift = quant_res, None
        table_name_to_quantized_weights[table_name] = (quant_weight, scale_shift)
    return device


def _get_device(module: nn.Module) -> torch.device:
    device = torch.device("cpu")

    for _, tensor in module.state_dict().items():
        device = tensor.device
        break
    return device


def _update_embedding_configs(
    embedding_configs: Sequence[BaseEmbeddingConfig],
    quant_config: Union[QuantConfig, torch.quantization.QConfig],
    tables_to_rows_post_pruning: Optional[Dict[str, int]] = None,
) -> None:
    per_table_weight_dtype = (
        quant_config.per_table_weight_dtype
        if isinstance(quant_config, QuantConfig) and quant_config.per_table_weight_dtype
        else {}
    )
    for config in embedding_configs:
        config.data_type = dtype_to_data_type(
            per_table_weight_dtype[config.name]
            if config.name in per_table_weight_dtype
            else quant_config.weight().dtype
        )

        if tables_to_rows_post_pruning and config.name in tables_to_rows_post_pruning:
            config.num_embeddings_post_pruning = tables_to_rows_post_pruning[
                config.name
            ]


@torch.fx.wrap
def _fx_trec_unwrap_kjt(
    kjt: KeyedJaggedTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forced conversions to support TBE
    CPU - int32 or int64, offsets dtype must match
    GPU - int32 only, offsets dtype must match
    """
    indices = kjt.values()
    offsets = kjt.offsets()
    if kjt.device().type == "cpu":
        return indices, offsets.type(dtype=indices.dtype)
    else:
        return indices.int(), offsets.int()


class EmbeddingBagCollection(EmbeddingBagCollectionInterface, ModuleNoCopyMixin):
    """
    This class represents a reimplemented version of the EmbeddingBagCollection
    class found in `torchrec/modules/embedding_modules.py`.
    However, it is quantized for lower precision.
    It relies on fbgemm quantized ops and provides table batching.

    For more details, including examples, please refer to
    `torchrec/modules/embedding_modules.py`
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool,
        device: torch.device,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
        register_tbes: bool = False,
        quant_state_dict_split_scale_bias: bool = False,
        row_alignment: int = DEFAULT_ROW_ALIGNMENT,
        cache_features_order: bool = False,
    ) -> None:
        super().__init__()
        self._is_weighted = is_weighted
        self._embedding_bag_configs: List[EmbeddingBagConfig] = tables
        self._key_to_tables: Dict[
            Tuple[PoolingType, bool], List[EmbeddingBagConfig]
        ] = defaultdict(list)
        self._feature_names: List[str] = []
        self._feature_splits: List[int] = []
        self._length_per_key: List[int] = []
        self._features_order: Optional[List[int]] = None
        # Registering in a List instead of ModuleList because we want don't want them to be auto-registered.
        # Their states will be modified via self.embedding_bags
        self._emb_modules: List[nn.Module] = []
        self._output_dtype = output_dtype
        self._device: torch.device = device
        self._table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None
        self.row_alignment = row_alignment
        self._kjt_to_jt_dict = ComputeKJTToJTDict()

        table_names = set()
        for table in self._embedding_bag_configs:
            if table.name in table_names:
                raise ValueError(f"Duplicate table name {table.name}")
            table_names.add(table.name)
            if hasattr(table, "use_virtual_table"):
                key = (table.pooling, table.use_virtual_table)
            else:
                key = (table.pooling, False)
            self._key_to_tables[key].append(table)

        location = (
            EmbeddingLocation.HOST if device.type == "cpu" else EmbeddingLocation.DEVICE
        )

        for (pooling, use_virtual_table), emb_configs in self._key_to_tables.items():
            embedding_specs = []
            weight_lists: Optional[
                List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            ] = ([] if table_name_to_quantized_weights else None)
            feature_table_map: List[int] = []

            for idx, table in enumerate(emb_configs):
                embedding_specs.append(
                    (
                        table.name,
                        (
                            table.num_embeddings_post_pruning
                            # TODO: Need to check if attribute exists for BC
                            if getattr(table, "num_embeddings_post_pruning", None)
                            is not None
                            else table.num_embeddings
                        ),
                        table.embedding_dim,
                        data_type_to_sparse_type(table.data_type),
                        location,
                    )
                )
                if table_name_to_quantized_weights:
                    none_throws(weight_lists).append(
                        table_name_to_quantized_weights[table.name]
                    )
                feature_table_map.extend([idx] * table.num_features())

            embedding_clazz = (
                KVEmbeddingInference
                if use_virtual_table
                else IntNBitTableBatchedEmbeddingBagsCodegen
            )
            emb_module = embedding_clazz(
                embedding_specs=embedding_specs,
                pooling_mode=pooling_type_to_pooling_mode(pooling),
                weight_lists=weight_lists,
                device=device,
                output_dtype=data_type_to_sparse_type(dtype_to_data_type(output_dtype)),
                row_alignment=row_alignment,
                feature_table_map=feature_table_map,
            )
            if weight_lists is None:
                emb_module.initialize_weights()
            self._emb_modules.append(emb_module)
            for table in emb_configs:
                self._feature_names.extend(table.feature_names)
            self._feature_splits.append(
                sum(table.num_features() for table in emb_configs)
            )

        ordered_tables = list(itertools.chain(*self._key_to_tables.values()))
        self._embedding_names: List[str] = list(
            itertools.chain(*get_embedding_names_by_table(ordered_tables))
        )
        for table in ordered_tables:
            self._length_per_key.extend(
                [table.embedding_dim] * len(table.feature_names)
            )

        # We map over the parameters from FBGEMM backed kernels to the canonical nn.EmbeddingBag
        # representation. This provides consistency between this class and the EmbeddingBagCollection
        # nn.Module API calls (state_dict, named_modules, etc)
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for (_key, tables), emb_module in zip(
            self._key_to_tables.items(), self._emb_modules
        ):
            for embedding_config, (weight, qscale, qbias) in zip(
                tables,
                # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
                emb_module.split_embedding_weights_with_scale_bias(
                    split_scale_bias_mode=2 if quant_state_dict_split_scale_bias else 0
                ),
            ):
                self.embedding_bags[embedding_config.name] = torch.nn.Module()
                # register as a buffer so it's exposed in state_dict.
                # TODO: register as param instead of buffer
                # however, since this is only needed for inference, we do not need to expose it as part of parameters.
                # Additionally, we cannot expose uint8 weights as parameters due to autograd restrictions.

                if embedding_config.data_type == DataType.INT4:
                    weight = UInt4Tensor(weight)
                elif embedding_config.data_type == DataType.INT2:
                    weight = UInt2Tensor(weight)

                self.embedding_bags[embedding_config.name].register_buffer(
                    "weight", weight
                )
                if quant_state_dict_split_scale_bias:
                    self.embedding_bags[embedding_config.name].register_buffer(
                        "weight_qscale", qscale
                    )
                    self.embedding_bags[embedding_config.name].register_buffer(
                        "weight_qbias", qbias
                    )

        setattr(
            self,
            MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
            quant_state_dict_split_scale_bias,
        )
        setattr(self, MODULE_ATTR_REGISTER_TBES_BOOL, register_tbes)
        self.register_tbes = register_tbes
        if register_tbes:
            self.tbes: torch.nn.ModuleList = torch.nn.ModuleList(self._emb_modules)
        setattr(self, MODULE_ATTR_CACHE_FEATURES_ORDER, cache_features_order)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """

        embeddings = []
        kjt_keys = _get_kjt_keys(features)
        # Cache the features order since the features will always have the same order of keys in inference.
        if getattr(self, MODULE_ATTR_CACHE_FEATURES_ORDER, False):
            if self._features_order is None:
                self._features_order = [kjt_keys.index(k) for k in self._feature_names]
                if self._features_order:
                    self.register_buffer(
                        "_features_order_tensor",
                        torch.tensor(
                            data=self._features_order,
                            device=features.device(),
                            dtype=torch.int32,
                        ),
                        persistent=False,
                    )
            kjt_permute = _permute_kjt(
                features,
                self._features_order,
                getattr(self, "_features_order_tensor", None),
            )
        else:
            kjt_permute_order = [kjt_keys.index(k) for k in self._feature_names]
            kjt_permute = _permute_kjt(features, kjt_permute_order)
        kjts_per_key = kjt_permute.split(self._feature_splits)

        for i, (emb_op, _) in enumerate(
            zip(self._emb_modules, self._key_to_tables.keys())
        ):
            f = kjts_per_key[i]
            indices, offsets = _fx_trec_unwrap_kjt(f)

            embeddings.append(
                # Syntax for FX to generate call_module instead of call_function to keep TBE copied unchanged to fx.GraphModule, can be done only for registered module
                emb_op(
                    indices=indices,
                    offsets=offsets,
                    per_sample_weights=f.weights() if self._is_weighted else None,
                )
                if self.register_tbes
                else emb_op.forward(
                    indices=indices,
                    offsets=offsets,
                    per_sample_weights=f.weights() if self._is_weighted else None,
                )
            )

        return KeyedTensor(
            keys=self._embedding_names,
            values=_cat_embeddings(embeddings),
            length_per_key=self._length_per_key,
        )

    def _get_name(self) -> str:
        return "QuantizedEmbeddingBagCollection"

    @classmethod
    def from_float(
        cls,
        module: OriginalEmbeddingBagCollection,
        use_precomputed_fake_quant: bool = False,
    ) -> "EmbeddingBagCollection":
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingBagCollection input float module must have qconfig defined"
        pruning_dict: Dict[str, int] = getattr(
            module, MODULE_ATTR_EMB_CONFIG_NAME_TO_NUM_ROWS_POST_PRUNING_DICT, {}
        )
        embedding_bag_configs = copy.deepcopy(module.embedding_bag_configs())
        _update_embedding_configs(
            cast(List[BaseEmbeddingConfig], embedding_bag_configs),
            # pyre-fixme[6]: For 2nd argument expected `Union[QuantConfig, QConfig]`
            #  but got `Union[Module, Tensor]`.
            module.qconfig,
            pruning_dict,
        )

        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        device = quantize_state_dict(
            module,
            table_name_to_quantized_weights,
            {table.name: table.data_type for table in embedding_bag_configs},
            pruning_dict,
        )
        return cls(
            embedding_bag_configs,
            module.is_weighted(),
            device=device,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `activation`.
            output_dtype=module.qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
            register_tbes=getattr(module, MODULE_ATTR_REGISTER_TBES_BOOL, False),
            quant_state_dict_split_scale_bias=getattr(
                module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            ),
            row_alignment=getattr(
                module, MODULE_ATTR_ROW_ALIGNMENT_INT, DEFAULT_ROW_ALIGNMENT
            ),
            cache_features_order=getattr(
                module, MODULE_ATTR_CACHE_FEATURES_ORDER, False
            ),
        )

    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def is_weighted(self) -> bool:
        return self._is_weighted

    def output_dtype(self) -> torch.dtype:
        return self._output_dtype

    @property
    def device(self) -> torch.device:
        return self._device


class FeatureProcessedEmbeddingBagCollection(EmbeddingBagCollection):
    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool,
        device: torch.device,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
        register_tbes: bool = False,
        quant_state_dict_split_scale_bias: bool = False,
        row_alignment: int = DEFAULT_ROW_ALIGNMENT,
        # feature processor is Optional only for the sake of the last position in constructor
        # Enforcing it to be non-None, for None case EmbeddingBagCollection must be used.
        feature_processor: Optional[FeatureProcessorsCollection] = None,
        cache_features_order: bool = False,
    ) -> None:
        super().__init__(
            tables,
            is_weighted,
            device,
            output_dtype,
            table_name_to_quantized_weights,
            register_tbes,
            quant_state_dict_split_scale_bias,
            row_alignment,
            cache_features_order,
        )
        assert (
            feature_processor is not None
        ), "Use EmbeddingBagCollection for no feature_processor"
        self.feature_processor: FeatureProcessorsCollection = feature_processor

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        features = self.feature_processor(features)
        return super().forward(features)

    def _get_name(self) -> str:
        return "QuantFeatureProcessedEmbeddingBagCollection"

    @classmethod
    # pyre-ignore
    def from_float(
        cls,
        module: OriginalFeatureProcessedEmbeddingBagCollection,
        use_precomputed_fake_quant: bool = False,
    ) -> "FeatureProcessedEmbeddingBagCollection":
        fp_ebc = module
        ebc = module._embedding_bag_collection
        qconfig = module.qconfig
        assert hasattr(
            module, "qconfig"
        ), "FeatureProcessedEmbeddingBagCollection input float module must have qconfig defined"

        pruning_dict: Dict[str, int] = getattr(
            module, MODULE_ATTR_EMB_CONFIG_NAME_TO_NUM_ROWS_POST_PRUNING_DICT, {}
        )

        embedding_bag_configs = copy.deepcopy(ebc.embedding_bag_configs())
        _update_embedding_configs(
            cast(List[BaseEmbeddingConfig], embedding_bag_configs),
            # pyre-fixme[6]: For 2nd argument expected `Union[QuantConfig, QConfig]`
            #  but got `Union[Module, Tensor]`.
            qconfig,
            pruning_dict,
        )

        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        device = quantize_state_dict(
            ebc,
            table_name_to_quantized_weights,
            {table.name: table.data_type for table in embedding_bag_configs},
            pruning_dict,
        )
        return cls(
            embedding_bag_configs,
            ebc.is_weighted(),
            device=device,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `activation`.
            output_dtype=qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
            register_tbes=getattr(module, MODULE_ATTR_REGISTER_TBES_BOOL, False),
            quant_state_dict_split_scale_bias=getattr(
                ebc, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            ),
            row_alignment=getattr(
                ebc, MODULE_ATTR_ROW_ALIGNMENT_INT, DEFAULT_ROW_ALIGNMENT
            ),
            # pyre-ignore
            feature_processor=fp_ebc._feature_processors,
            cache_features_order=getattr(
                module, MODULE_ATTR_CACHE_FEATURES_ORDER, False
            ),
        )


class EmbeddingCollection(EmbeddingCollectionInterface, ModuleNoCopyMixin):
    """
    This class represents a reimplemented version of the EmbeddingCollection
    class found in `torchrec/modules/embedding_modules.py`.
    However, it is quantized for lower precision.
    It relies on fbgemm quantized ops and provides table batching.

    For more details, including examples, please refer to
    `torchrec/modules/embedding_modules.py`
    """

    def __init__(  # noqa C901
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        need_indices: bool = False,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
        register_tbes: bool = False,
        quant_state_dict_split_scale_bias: bool = False,
        row_alignment: int = DEFAULT_ROW_ALIGNMENT,
        cache_features_order: bool = False,
    ) -> None:
        super().__init__()
        self._emb_modules: List[IntNBitTableBatchedEmbeddingBagsCodegen] = []
        self._embedding_configs = tables
        self._embedding_dim: int = -1
        self._need_indices: bool = need_indices
        self._output_dtype = output_dtype
        self._device = device
        self.row_alignment = row_alignment
        self._key_to_tables: Dict[Tuple[DataType, bool], List[EmbeddingConfig]] = (
            defaultdict(list)
        )
        self._feature_names: List[str] = []
        self._features_order: Optional[List[int]] = None

        self._table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = table_name_to_quantized_weights

        table_names = set()
        for table in self._embedding_configs:
            if table.name in table_names:
                raise ValueError(f"Duplicate table name {table.name}")
            table_names.add(table.name)
            self._embedding_dim = (
                table.embedding_dim if self._embedding_dim < 0 else self._embedding_dim
            )
            if self._embedding_dim != table.embedding_dim:
                raise ValueError(
                    "All tables in a EmbeddingCollection are required to have same embedding dimension."
                    + f" Violating case: {table.name}'s embedding_dim {table.embedding_dim} !="
                    + f" {self._embedding_dim}"
                )
            if hasattr(table, "use_virtual_table"):
                key = (table.data_type, table.use_virtual_table)
            else:
                key = (table.data_type, False)
            self._key_to_tables[key].append(table)
        self._feature_splits: List[int] = []
        for (data_type, use_virtual_table), emb_configs in self._key_to_tables.items():
            embedding_specs = []
            weight_lists: Optional[
                List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            ] = ([] if table_name_to_quantized_weights else None)
            feature_table_map: List[int] = []
            for idx, table in enumerate(emb_configs):
                embedding_specs.append(
                    (
                        table.name,
                        table.num_embeddings,
                        table.embedding_dim,
                        data_type_to_sparse_type(data_type),
                        (
                            EmbeddingLocation.HOST
                            if device.type == "cpu"
                            else EmbeddingLocation.DEVICE
                        ),
                    )
                )
                if table_name_to_quantized_weights:
                    none_throws(weight_lists).append(
                        table_name_to_quantized_weights[table.name]
                    )
                feature_table_map.extend([idx] * table.num_features())
                self._feature_names.extend(table.feature_names)
            embedding_clazz = (
                KVEmbeddingInference
                if use_virtual_table
                else IntNBitTableBatchedEmbeddingBagsCodegen
            )
            emb_module = embedding_clazz(
                embedding_specs=embedding_specs,
                pooling_mode=PoolingMode.NONE,
                weight_lists=weight_lists,
                device=device,
                output_dtype=data_type_to_sparse_type(dtype_to_data_type(output_dtype)),
                row_alignment=row_alignment,
                feature_table_map=feature_table_map,
            )
            if weight_lists is None:
                emb_module.initialize_weights()
            self._emb_modules.append(emb_module)
            self._feature_splits.append(
                sum(table.num_features() for table in emb_configs)
            )

        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        for (_key, tables), emb_module in zip(
            self._key_to_tables.items(), self._emb_modules
        ):
            for embedding_config, (weight, qscale, qbias) in zip(
                tables,
                emb_module.split_embedding_weights_with_scale_bias(
                    split_scale_bias_mode=2 if quant_state_dict_split_scale_bias else 0
                ),
            ):
                self.embeddings[embedding_config.name] = torch.nn.Module()
                # register as a buffer so it's exposed in state_dict.
                # TODO: register as param instead of buffer
                # however, since this is only needed for inference, we do not need to expose it as part of parameters.
                # Additionally, we cannot expose uint8 weights as parameters due to autograd restrictions.
                if embedding_config.data_type == DataType.INT4:
                    weight = UInt4Tensor(weight)
                elif embedding_config.data_type == DataType.INT2:
                    weight = UInt2Tensor(weight)
                self.embeddings[embedding_config.name].register_buffer("weight", weight)
                if quant_state_dict_split_scale_bias:
                    self.embeddings[embedding_config.name].register_buffer(
                        "weight_qscale", qscale
                    )
                    self.embeddings[embedding_config.name].register_buffer(
                        "weight_qbias", qbias
                    )

        self._embedding_names_by_batched_tables: Dict[
            Tuple[DataType, bool], List[str]
        ] = {
            key: list(itertools.chain(*get_embedding_names_by_table(table)))
            for key, table in self._key_to_tables.items()
        }

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            self._embedding_configs
        )
        setattr(
            self,
            MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
            quant_state_dict_split_scale_bias,
        )
        setattr(self, MODULE_ATTR_REGISTER_TBES_BOOL, register_tbes)
        self.register_tbes = register_tbes
        if register_tbes:
            self.tbes: torch.nn.ModuleList = torch.nn.ModuleList(self._emb_modules)
        setattr(self, MODULE_ATTR_CACHE_FEATURES_ORDER, cache_features_order)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """

        feature_embeddings: Dict[str, JaggedTensor] = {}
        kjt_keys = _get_kjt_keys(features)
        # Cache the features order since the features will always have the same order of keys in inference.
        if getattr(self, MODULE_ATTR_CACHE_FEATURES_ORDER, False):
            if self._features_order is None:
                self._features_order = [kjt_keys.index(k) for k in self._feature_names]
                if self._features_order:
                    self.register_buffer(
                        "_features_order_tensor",
                        torch.tensor(
                            data=self._features_order,
                            device=features.device(),
                            dtype=torch.int32,
                        ),
                        persistent=False,
                    )
            kjt_permute = _permute_kjt(
                features,
                self._features_order,
                getattr(self, "_features_order_tensor", None),
            )
        else:
            kjt_permute_order = [kjt_keys.index(k) for k in self._feature_names]
            kjt_permute = _permute_kjt(features, kjt_permute_order)
        kjts_per_key = kjt_permute.split(self._feature_splits)

        for i, (emb_module, key) in enumerate(
            zip(self._emb_modules, self._key_to_tables.keys())
        ):
            f = kjts_per_key[i]
            lengths = _get_feature_length(f)
            indices, offsets = _fx_trec_unwrap_kjt(f)
            embedding_names = self._embedding_names_by_batched_tables[key]
            lookup = (
                emb_module(indices=indices, offsets=offsets)
                if self.register_tbes
                else emb_module.forward(indices=indices, offsets=offsets)
            )
            if getattr(self, MODULE_ATTR_USE_UNFLATTENED_LENGTHS_FOR_BATCHING, False):
                lengths = _get_unflattened_lengths(lengths, len(embedding_names))
                lookup = _get_batching_hinted_output(lengths=lengths, output=lookup)
            else:
                if getattr(self, MODULE_ATTR_USE_BATCHING_HINTED_OUTPUT, True):
                    lookup = _get_batching_hinted_output(lengths=lengths, output=lookup)
                lengths = _get_unflattened_lengths(lengths, len(embedding_names))
            jt = construct_jagged_tensors_inference(
                embeddings=lookup,
                lengths=lengths,
                values=indices,
                embedding_names=embedding_names,
                need_indices=self.need_indices(),
                remove_padding=getattr(
                    self, MODULE_ATTR_REMOVE_STBE_PADDING_BOOL, False
                ),
            )
            for embedding_name in embedding_names:
                feature_embeddings[embedding_name] = jt[embedding_name]
        return feature_embeddings

    @classmethod
    def from_float(
        cls,
        module: OriginalEmbeddingCollection,
        use_precomputed_fake_quant: bool = False,
    ) -> "EmbeddingCollection":
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingCollection input float module must have qconfig defined"
        embedding_configs = copy.deepcopy(module.embedding_configs())
        _update_embedding_configs(
            cast(List[BaseEmbeddingConfig], embedding_configs),
            # pyre-fixme[6]: For 2nd argument expected `Union[QuantConfig, QConfig]`
            #  but got `Union[Module, Tensor]`.
            module.qconfig,
        )
        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        device = quantize_state_dict(
            module,
            table_name_to_quantized_weights,
            {table.name: table.data_type for table in embedding_configs},
        )
        return cls(
            embedding_configs,
            device=device,
            need_indices=module.need_indices(),
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `activation`.
            output_dtype=module.qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
            register_tbes=getattr(module, MODULE_ATTR_REGISTER_TBES_BOOL, False),
            quant_state_dict_split_scale_bias=getattr(
                module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            ),
            row_alignment=getattr(
                module, MODULE_ATTR_ROW_ALIGNMENT_INT, DEFAULT_ROW_ALIGNMENT
            ),
            cache_features_order=getattr(
                module, MODULE_ATTR_CACHE_FEATURES_ORDER, False
            ),
        )

    def _get_name(self) -> str:
        return "QuantizedEmbeddingCollection"

    def need_indices(self) -> bool:
        return self._need_indices

    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_configs

    def embedding_names_by_table(self) -> List[List[str]]:
        return self._embedding_names_by_table

    def output_dtype(self) -> torch.dtype:
        return self._output_dtype

    @property
    def device(self) -> torch.device:
        return self._device


class QuantManagedCollisionEmbeddingCollection(EmbeddingCollection):
    """
    QuantManagedCollisionEmbeddingCollection represents a quantized EC module and a set of managed collision modules.
    The inputs into the MC-EC/EBC will first be modified by the managed collision module before being passed into the embedding collection.

    Args:
        tables (List[EmbeddingConfig]): A list of EmbeddingConfig objects representing the embedding tables in the collection.
        device (torch.device): The device on which the embedding collection will be allocated.
        need_indices (bool, optional): Whether to return the indices along with the embeddings. Defaults to False.
        output_dtype (torch.dtype, optional): The data type of the output embeddings. Defaults to torch.float.
        table_name_to_quantized_weights (Dict[str, Tuple[Tensor, Tensor]], optional): A dictionary mapping table names to their corresponding quantized weights. Defaults to None.
        register_tbes (bool, optional): Whether to register the TBEs in the model. Defaults to False.
        quant_state_dict_split_scale_bias (bool, optional): Whether to split the scale and bias parameters when saving the quantized state dict. Defaults to False.
        row_alignment (int, optional): The alignment of rows in the quantized weights. Defaults to DEFAULT_ROW_ALIGNMENT.
        managed_collision_collection (ManagedCollisionCollection, optional): The managed collision collection to use for managing collisions. Defaults to None.
        return_remapped_features (bool, optional): Whether to return the remapped input features in addition to the embeddings. Defaults to False.
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        need_indices: bool = False,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
        register_tbes: bool = False,
        quant_state_dict_split_scale_bias: bool = False,
        row_alignment: int = DEFAULT_ROW_ALIGNMENT,
        managed_collision_collection: Optional[ManagedCollisionCollection] = None,
        return_remapped_features: bool = False,
        cache_features_order: bool = False,
    ) -> None:
        super().__init__(
            tables,
            device,
            need_indices,
            output_dtype,
            table_name_to_quantized_weights,
            register_tbes,
            quant_state_dict_split_scale_bias,
            row_alignment,
            cache_features_order,
        )
        assert (
            managed_collision_collection
        ), "Managed collision collection cannot be None"
        self._managed_collision_collection: ManagedCollisionCollection = (
            managed_collision_collection
        )
        self._return_remapped_features = return_remapped_features

        assert str(self.embedding_configs()) == str(
            self._managed_collision_collection.embedding_configs()
        ), "Embedding Collection and Managed Collision Collection must contain the same Embedding Configs"

        # Assuming quantized MCEC is used in inference only
        for (
            managed_collision_module
        ) in self._managed_collision_collection._managed_collision_modules.values():
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            managed_collision_module.reset_inference_mode()

    def to(
        self, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> "QuantManagedCollisionEmbeddingCollection":
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(
            *args,  # pyre-ignore
            **kwargs,  # pyre-ignore
        )
        for param in self.parameters():
            if param.device.type != "meta":
                param.to(device)

        for buffer in self.buffers():
            if buffer.device.type != "meta":
                buffer.to(device)
        # Skip device movement and continue with other args
        super().to(
            dtype=dtype,
            non_blocking=non_blocking,
        )
        return self

    # pyre-ignore
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:
        features = self._managed_collision_collection(features)

        return (super().forward(features), features)

    def _get_name(self) -> str:
        return "QuantManagedCollisionEmbeddingCollection"

    @classmethod
    # pyre-ignore
    def from_float(
        cls,
        module: OriginalManagedCollisionEmbeddingCollection,
        return_remapped_features: bool = False,
    ) -> "QuantManagedCollisionEmbeddingCollection":
        mc_ec = module
        ec = module._embedding_module

        # pyre-ignore[9]
        qconfig: torch.quantization.QConfig = module.qconfig
        assert hasattr(
            module, "qconfig"
        ), "QuantManagedCollisionEmbeddingCollection input float module must have qconfig defined"

        # pyre-ignore[29]
        embedding_configs = copy.deepcopy(ec.embedding_configs())
        _update_embedding_configs(
            cast(List[BaseEmbeddingConfig], embedding_configs),
            qconfig,
        )
        _update_embedding_configs(
            mc_ec._managed_collision_collection._embedding_configs,
            qconfig,
        )

        # pyre-ignore[9]
        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] | None = (
            ec._table_name_to_quantized_weights
            if hasattr(ec, "_table_name_to_quantized_weights")
            else None
        )
        device = _get_device(ec)
        return cls(
            embedding_configs,
            device=device,
            output_dtype=qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
            register_tbes=getattr(module, MODULE_ATTR_REGISTER_TBES_BOOL, False),
            quant_state_dict_split_scale_bias=getattr(
                ec, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            ),
            row_alignment=getattr(
                ec, MODULE_ATTR_ROW_ALIGNMENT_INT, DEFAULT_ROW_ALIGNMENT
            ),
            managed_collision_collection=mc_ec._managed_collision_collection,
            return_remapped_features=mc_ec._return_remapped_features,
            cache_features_order=getattr(ec, MODULE_ATTR_CACHE_FEATURES_ORDER, False),
        )
