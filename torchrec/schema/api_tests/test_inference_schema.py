#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import inspect
import unittest
from typing import Any, cast, Dict, List, Optional, Tuple, Type

import torch
from torchrec.distributed.fused_params import (
    FUSED_PARAM_BOUNDS_CHECK_MODE,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
)
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollection,
    QuantEmbeddingCollectionSharder,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    QuantEmbeddingBagCollectionSharder,
    QuantFeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.types import BoundsCheckMode, ModuleSharder, ShardingPlan
from torchrec.inference.modules import (
    DEFAULT_FUSED_PARAMS,
    DEFAULT_QUANT_MAPPING,
    DEFAULT_QUANTIZATION_DTYPE,
    DEFAULT_SHARDERS,
    quantize_inference_model,
    shard_quant_model,
    trim_torch_package_prefix_from_typename,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.schema.utils import is_signature_compatible

STABLE_DEFAULT_QUANTIZATION_DTYPE: torch.dtype = torch.int8


STABLE_DEFAULT_FUSED_PARAMS: Dict[str, Any] = {
    FUSED_PARAM_REGISTER_TBE_BOOL: True,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: True,
    FUSED_PARAM_BOUNDS_CHECK_MODE: BoundsCheckMode.NONE,
}

STABLE_DEFAULT_SHARDERS: List[ModuleSharder[torch.nn.Module]] = [
    cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingBagCollectionSharder(fused_params=STABLE_DEFAULT_FUSED_PARAMS),
    ),
    cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingCollectionSharder(fused_params=STABLE_DEFAULT_FUSED_PARAMS),
    ),
    cast(
        ModuleSharder[torch.nn.Module],
        QuantFeatureProcessedEmbeddingBagCollectionSharder(
            fused_params=STABLE_DEFAULT_FUSED_PARAMS
        ),
    ),
]

STABLE_DEFAULT_QUANT_MAPPING: Dict[str, Type[torch.nn.Module]] = {
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingBagCollection)
    ): QuantEmbeddingBagCollection,
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingCollection)
    ): QuantEmbeddingCollection,
}


def stable_quantize_inference_model(
    model: torch.nn.Module,
    quantization_mapping: Optional[Dict[str, Type[torch.nn.Module]]] = None,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
    fp_weight_dtype: torch.dtype = STABLE_DEFAULT_QUANTIZATION_DTYPE,
    quantization_dtype: torch.dtype = STABLE_DEFAULT_QUANTIZATION_DTYPE,
    output_dtype: torch.dtype = torch.float,
) -> torch.nn.Module:
    return model


def stable_shard_quant_model(
    model: torch.nn.Module,
    world_size: int = 1,
    compute_device: str = "cuda",
    sharding_device: str = "meta",
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    device_memory_size: Optional[int] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
) -> Tuple[torch.nn.Module, ShardingPlan]:
    return (model, ShardingPlan(plan={}))


class TestInferenceSchema(unittest.TestCase):
    def test_quantize_inference_model(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(stable_quantize_inference_model),
                inspect.signature(quantize_inference_model),
            )
        )

    def test_shard_quant_model(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(stable_shard_quant_model),
                inspect.signature(shard_quant_model),
            )
        )

    def test_default_mappings(self) -> None:
        # check that the default mappings are a superset of the stable ones
        for (
            name,
            module_type,
        ) in STABLE_DEFAULT_QUANT_MAPPING.items():
            self.assertTrue(name in DEFAULT_QUANT_MAPPING)
            self.assertTrue(DEFAULT_QUANT_MAPPING[name] == module_type)

        # check that the fused params are a superset of the stable ones
        for (
            name,
            val,
        ) in STABLE_DEFAULT_FUSED_PARAMS.items():
            self.assertTrue(name in DEFAULT_FUSED_PARAMS)
            self.assertTrue(DEFAULT_FUSED_PARAMS[name] == val)

        # Check default quant type
        self.assertTrue(DEFAULT_QUANTIZATION_DTYPE == STABLE_DEFAULT_QUANTIZATION_DTYPE)

        # Check default sharders are a superset of the stable ones
        # and check fused_params are also a superset
        for sharder in STABLE_DEFAULT_SHARDERS:
            found = False
            for default_sharder in DEFAULT_SHARDERS:
                if isinstance(default_sharder, type(sharder)):
                    for key in sharder.fused_params.keys():
                        self.assertTrue(key in default_sharder.fused_params)
                        self.assertTrue(
                            default_sharder.fused_params[key]
                            == sharder.fused_params[key]
                        )
                    found = True

            self.assertTrue(found)
