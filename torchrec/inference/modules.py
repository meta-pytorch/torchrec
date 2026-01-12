#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.quantization as quant
import torchrec as trec
import torchrec.distributed as trec_dist
import torchrec.quant as trec_quant
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch.fx.passes.split_utils import getattr_recursive
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fused_params import (
    FUSED_PARAM_BOUNDS_CHECK_MODE,
    FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
)
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
)
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollectionSharder,
    QuantFeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.types import (
    BoundsCheckMode,
    ModuleSharder,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import QuantConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
    FeatureProcessedEmbeddingBagCollection as QuantFeatureProcessedEmbeddingBagCollection,
    MODULE_ATTR_EMB_CONFIG_NAME_TO_NUM_ROWS_POST_PRUNING_DICT,
    quant_prep_enable_cache_features_order,
    quant_prep_enable_register_tbes,
)

logger: logging.Logger = logging.getLogger(__name__)


def trim_torch_package_prefix_from_typename(typename: str) -> str:
    if typename.startswith("<torch_package_"):
        # Trim off <torch_package_x> prefix.
        typename = ".".join(typename.split(".")[1:])
    return typename


DEFAULT_FUSED_PARAMS: Dict[str, Any] = {
    FUSED_PARAM_REGISTER_TBE_BOOL: True,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: True,
    FUSED_PARAM_BOUNDS_CHECK_MODE: BoundsCheckMode.NONE,
    FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP: False,
}

DEFAULT_SHARDERS: List[ModuleSharder[torch.nn.Module]] = [
    cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingBagCollectionSharder(fused_params=DEFAULT_FUSED_PARAMS),
    ),
    cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingCollectionSharder(fused_params=DEFAULT_FUSED_PARAMS),
    ),
    cast(
        ModuleSharder[torch.nn.Module],
        QuantFeatureProcessedEmbeddingBagCollectionSharder(
            fused_params=DEFAULT_FUSED_PARAMS
        ),
    ),
]

DEFAULT_QUANT_MAPPING: Dict[str, Type[torch.nn.Module]] = {
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingBagCollection)
    ): QuantEmbeddingBagCollection,
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingCollection)
    ): QuantEmbeddingCollection,
}

DEFAULT_QUANTIZATION_DTYPE: torch.dtype = torch.int8

FEATURE_PROCESSED_EBC_TYPE: str = trim_torch_package_prefix_from_typename(
    torch.typename(FeatureProcessedEmbeddingBagCollection)
)


def quantize_feature(
    module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        [
            (
                input.half()
                if isinstance(input, torch.Tensor)
                and input.dtype in [torch.float32, torch.float64]
                else input
            )
            for input in inputs
        ]
    )


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
    output_dtype: torch.dtype = torch.float,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
) -> nn.Module:
    qconfig = QuantConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
        per_table_weight_dtype=per_table_weight_dtype,
    )
    qconfig_spec: Dict[Type[nn.Module], QuantConfig] = {
        trec.EmbeddingBagCollection: qconfig,
    }
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {
        trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
    }
    if additional_qconfig_spec_keys is not None:
        for t in additional_qconfig_spec_keys:
            qconfig_spec[t] = qconfig
    if additional_mapping is not None:
        mapping.update(additional_mapping)
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
        inplace=inplace,
    )


@dataclass
class QualNameMetadata:
    need_preproc: bool


@dataclass
class BatchingMetadata:
    """
    Metadata class for batching, this should be kept in sync with the C++ definition.
    """

    type: str
    # cpu or cuda
    device: str
    # list of tensor suffixes to deserialize to pinned memory (e.g. "lengths")
    # use "" (empty string) to pin without suffix
    pinned: List[str]


class PredictFactory(abc.ABC):
    """
    Creates a model (with already learned weights) to be used inference time.
    """

    @abc.abstractmethod
    def create_predict_module(self) -> nn.Module:
        """
        Returns already sharded model with allocated weights.
        state_dict() must match TransformModule.transform_state_dict().
        It assumes that torch.distributed.init_process_group was already called
        and will shard model according to torch.distributed.get_world_size().
        """
        pass

    @abc.abstractmethod
    def batching_metadata(self) -> Dict[str, BatchingMetadata]:
        """
        Returns a dict from input name to BatchingMetadata. This infomation is used for batching for input requests.
        """
        pass

    def batching_metadata_json(self) -> str:
        """
        Serialize the batching metadata to JSON, for ease of parsing with torch::deploy environments.
        """
        return json.dumps(
            {key: asdict(value) for key, value in self.batching_metadata().items()}
        )

    @abc.abstractmethod
    def result_metadata(self) -> str:
        """
        Returns a string which represents the result type. This information is used for result split.
        """
        pass

    @abc.abstractmethod
    def run_weights_independent_tranformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that don't rely on weights of the predict module. e.g. fx tracing, model
        split etc.
        """
        pass

    @abc.abstractmethod
    def run_weights_dependent_transformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that depends on weights of the predict module. e.g. lowering to a backend.
        """
        pass

    def qualname_metadata(self) -> Dict[str, QualNameMetadata]:
        """
        Returns a dict from qualname (method name) to QualNameMetadata. This is additional information for execution of specific methods of the model.
        """
        return {}

    def qualname_metadata_json(self) -> str:
        """
        Serialize the qualname metadata to JSON, for ease of parsing with torch::deploy environments.
        """
        return json.dumps(
            {key: asdict(value) for key, value in self.qualname_metadata().items()}
        )

    def model_inputs_data(self) -> Dict[str, Any]:
        """
        Returns a dict of various data for benchmarking input generation.
        """
        return {}


class PredictModule(nn.Module):
    """
    Interface for modules to work in a torch.deploy based backend. Users should
    override predict_forward to convert batch input format to module input format.

    Call Args:
        batch: a dict of input tensors

    Returns:
        output: a dict of output tensors

    Args:
        module: the actual predict module
        device: the primary device for this module that will be used in forward calls.

    Example::

        module = PredictModule(torch.device("cuda", torch.cuda.current_device()))
    """

    def __init__(
        self,
        module: nn.Module,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._module: nn.Module = module
        # lazy init device from thread inited device guard
        self._device: Optional[torch.device] = torch.device(device) if device else None
        self._module.eval()

    @property
    def predict_module(
        self,
    ) -> nn.Module:
        return self._module

    @abc.abstractmethod
    # pyre-fixme[3]
    def predict_forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        pass

    # pyre-fixme[3]
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        if self._device is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        with torch.inference_mode():
            return self.predict_forward(batch)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        # pyre-fixme[19]: Expected 0 positional arguments.
        return self._module.state_dict(destination, prefix, keep_vars)


def quantize_dense(
    predict_module: PredictModule,
    dtype: torch.dtype,
    additional_embedding_module_type: List[Type[nn.Module]] = [],
) -> nn.Module:
    module = predict_module.predict_module
    reassign = {}

    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not (
            isinstance(mod, EmbeddingBagCollectionInterface)
            or isinstance(mod, EmbeddingCollectionInterface)
            or any([type(mod) is clazz for clazz in additional_embedding_module_type])
        ):
            if dtype == torch.half:
                new_mod = mod.half()
                new_mod.register_forward_pre_hook(quantize_feature)
                reassign[name] = new_mod
            else:
                raise NotImplementedError(
                    "only fp16 is supported for non-embedding module lowering"
                )
    for key, value in reassign.items():
        module._modules[key] = value
    return predict_module


def set_pruning_data(
    model: torch.nn.Module,
    tables_to_rows_post_pruning: Dict[str, int],
    module_types: Optional[List[Type[nn.Module]]] = None,
) -> torch.nn.Module:
    if module_types is None:
        module_types = [EmbeddingBagCollection, FeatureProcessedEmbeddingBagCollection]

    for _, module in model.named_modules():
        if type(module) in module_types:
            setattr(
                module,
                MODULE_ATTR_EMB_CONFIG_NAME_TO_NUM_ROWS_POST_PRUNING_DICT,
                tables_to_rows_post_pruning,
            )

    return model


def quantize_inference_model(
    model: torch.nn.Module,
    quantization_mapping: Optional[Dict[str, Type[torch.nn.Module]]] = None,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
    fp_weight_dtype: torch.dtype = DEFAULT_QUANTIZATION_DTYPE,
    quantization_dtype: torch.dtype = DEFAULT_QUANTIZATION_DTYPE,
    output_dtype: torch.dtype = torch.float,
) -> torch.nn.Module:
    """
    Quantize the model, module swapping TorchRec train modules with its
    quantized counterpart, (e.g. EmbeddingBagCollection -> QuantEmbeddingBagCollection).

    Args:
        model (torch.nn.Module): the model to be quantized
        quantization_mapping (Optional[Dict[str, Type[torch.nn.Module]]]): a mapping from
            the original module type to the quantized module type. If not provided, the default mapping will be used:
            (EmbeddingBagCollection -> QuantEmbeddingBagCollection, EmbeddingCollection -> QuantEmbeddingCollection).
        per_table_weight_dtype (Optional[Dict[str, torch.dtype]]): a mapping from table name to weight dtype.
            If not provided, the default quantization dtype will be used (int8).
        fp_weight_dtype (torch.dtype): the desired quantized dtype for feature processor weights in
            FeatureProcessedEmbeddingBagCollection if used. Default is int8.

    Returns:
        torch.nn.Module: the quantized model

    Example::

        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            id_list_features_keys=self.model_config.id_list_features_keys,
            dense_device=device,
        )

        quant_model = quantize_inference_model(module)
    """

    if quantization_mapping is None:
        quantization_mapping = DEFAULT_QUANT_MAPPING

    def _quantize_fp_module(
        model: torch.nn.Module,
        fp_module: FeatureProcessedEmbeddingBagCollection,
        fp_module_fqn: str,
        weight_dtype: torch.dtype = DEFAULT_QUANTIZATION_DTYPE,
        per_fp_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
    ) -> None:
        """
        If FeatureProcessedEmbeddingBagCollection is found, quantize via direct module swap.
        """

        quant_prep_enable_register_tbes(model, [FeatureProcessedEmbeddingBagCollection])
        quant_prep_enable_cache_features_order(
            model, [FeatureProcessedEmbeddingBagCollection]
        )
        # pyre-fixme[16]: `FeatureProcessedEmbeddingBagCollection` has no attribute
        #  `qconfig`.
        fp_module.qconfig = QuantConfig(
            activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
            weight=quant.PlaceholderObserver.with_args(dtype=weight_dtype),
            per_table_weight_dtype=per_fp_table_weight_dtype,
        )

        # ie. "root.submodule.feature_processed_mod" -> "root.submodule", "feature_processed_mod"
        fp_ebc_parent_fqn, fp_ebc_name = fp_module_fqn.rsplit(".", 1)
        fp_ebc_parent = getattr_recursive(model, fp_ebc_parent_fqn)
        fp_ebc_parent.register_module(
            fp_ebc_name,
            QuantFeatureProcessedEmbeddingBagCollection.from_float(fp_module),
        )

    additional_qconfig_spec_keys = []
    additional_mapping = {}

    for n, m in model.named_modules():
        typename = trim_torch_package_prefix_from_typename(torch.typename(m))

        if typename in quantization_mapping:
            additional_qconfig_spec_keys.append(type(m))
            additional_mapping[type(m)] = quantization_mapping[typename]
        elif typename == FEATURE_PROCESSED_EBC_TYPE:
            # handle the fp ebc separately
            _quantize_fp_module(
                model,
                m,
                n,
                weight_dtype=fp_weight_dtype,
                # Pass in per_fp_table_weight_dtype if it is provided, perhaps
                # fpebc parameters are also in here
                per_fp_table_weight_dtype=per_table_weight_dtype,
            )

    quant_prep_enable_register_tbes(model, list(additional_mapping.keys()))
    quant_prep_enable_cache_features_order(
        model, [EmbeddingBagCollection, EmbeddingCollection]
    )
    quantize_embeddings(
        model,
        dtype=quantization_dtype,
        additional_qconfig_spec_keys=additional_qconfig_spec_keys,
        additional_mapping=additional_mapping,
        inplace=True,
        per_table_weight_dtype=per_table_weight_dtype,
        output_dtype=output_dtype,
    )

    logger.info(
        f"Default quantization dtype is {quantization_dtype}, {per_table_weight_dtype=}."
    )

    return model


def shard_quant_model(
    model: torch.nn.Module,
    world_size: int = 1,
    compute_device: str = "cuda",
    sharding_device: str = "meta",
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    device_memory_size: Optional[int] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ddr_cap: Optional[int] = None,
    sharding_plan: Optional[ShardingPlan] = None,
) -> Tuple[torch.nn.Module, ShardingPlan]:
    """
    Shard a quantized TorchRec model, used for generating the most optimal model for inference and
    necessary for distributed inference.

    Args:
        model (torch.nn.Module): the quantized model to be sharded
        world_size (int): the number of devices to shard the model, default to 1
        compute_device (str): the device to run the model, default to "cuda"
        sharding_device (str): the device to run the sharding, default to "meta"
        sharders (Optional[List[ModuleSharder[torch.nn.Module]]]): sharders to use for sharding
            quantized model, default to QuantEmbeddingBagCollectionSharder, QuantEmbeddingCollectionSharder,
            QuantFeatureProcessedEmbeddingBagCollectionSharder.
        device_memory_size (Optional[int]): the memory limit for cuda devices, default to None
        constraints (Optional[Dict[str, ParameterConstraints]]): constraints to use for sharding, default to None
            which will then implement default constraints with QuantEmbeddingBagCollection being sharded TableWise

    Returns:
        Tuple[torch.nn.Module, ShardingPlan]: the sharded model and the sharding plan

    Example::
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            id_list_features_keys=self.model_config.id_list_features_keys,
            dense_device=device,
        )

        quant_model = quantize_inference_model(module)
        sharded_model, _ = shard_quant_model(quant_model)
    """

    if sharding_plan is not None:
        model = _shard_modules(
            module=model,
            device=torch.device(sharding_device),
            plan=sharding_plan,
            env=trec_dist.ShardingEnv.from_local(
                world_size,
                0,
            ),
            sharders=sharders if sharders else DEFAULT_SHARDERS,
        )

        return model, sharding_plan

    if constraints is None:
        table_fqns = []
        sharders = sharders if sharders else DEFAULT_SHARDERS
        module_types = [sharder.module_type for sharder in sharders]
        for module in model.modules():
            if type(module) in module_types:
                # TODO: handle other cases/reduce hardcoding
                if hasattr(module, "embedding_bags"):
                    # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Module, Tensor]`
                    #  is not a function.
                    for table in module.embedding_bags:
                        table_fqns.append(table)

        # Default table wise constraints
        constraints = {}
        for name in table_fqns:
            constraints[name] = ParameterConstraints(
                sharding_types=[ShardingType.TABLE_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
            )

    if device_memory_size is not None:
        hbm_cap = device_memory_size
    elif torch.cuda.is_available() and compute_device == "cuda":
        hbm_cap = torch.cuda.get_device_properties(
            f"cuda:{torch.cuda.current_device()}"
        ).total_memory
    else:
        hbm_cap = None

    topology = trec_dist.planner.Topology(
        world_size=world_size,
        compute_device=compute_device,
        local_world_size=world_size,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,
    )
    batch_size = 1
    model_plan = trec_dist.planner.EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        constraints=constraints,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            constraints=constraints,
            estimator=[
                EmbeddingPerfEstimator(
                    topology=topology, constraints=constraints, is_inference=True
                ),
                EmbeddingStorageEstimator(topology=topology, constraints=constraints),
            ],
        ),
        storage_reservation=FixedPercentageStorageReservation(
            percentage=0.0,
        ),
    ).plan(
        model,
        sharders if sharders else DEFAULT_SHARDERS,
    )

    model = _shard_modules(
        module=model,
        device=torch.device(sharding_device),
        plan=model_plan,
        env=trec_dist.ShardingEnv.from_local(
            world_size,
            0,
        ),
        sharders=sharders if sharders else DEFAULT_SHARDERS,
    )

    return model, model_plan


def get_table_to_weights_from_tbe(
    model: torch.nn.Module,
) -> Dict[str, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    table_to_weight = {}

    for module in model.modules():
        if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen):
            weights = module.split_embedding_weights()
            for i, spec in enumerate(module.embedding_specs):
                table_to_weight[spec[0]] = weights[i]

    return table_to_weight


def assign_weights_to_tbe(
    model: torch.nn.Module,
    table_to_weight: Dict[str, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]],
) -> None:
    for module in model.modules():
        if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen):
            q_weights = []
            for spec in module.embedding_specs:
                assert spec[0] in table_to_weight, f"{spec[0]} not in table_to_weight"
                q_weights.append(table_to_weight[spec[0]])

            module.assign_embedding_weights(q_weights)

    return
