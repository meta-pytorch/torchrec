#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Iterable, Optional

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
from torchrec.distributed.types import BoundsCheckMode

FUSED_PARAM_REGISTER_TBE_BOOL: str = "__register_tbes_in_named_modules"
FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: str = (
    "__register_quant_state_dict_split_scale_bias"
)
FUSED_PARAM_TBE_ROW_ALIGNMENT: str = "__register_tbe_row_alignment"
FUSED_PARAM_BOUNDS_CHECK_MODE: str = "__register_tbe_bounds_check_mode"

# Force lengths to offsets conversion before TBE lookup. Helps with performance
# with certain ways to split models.
FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP: str = "__register_lengths_to_offsets_lookup"

# Fused param storing list of cpu embedding tables offloaded to ssd to scale
# the embedding table size
FUSED_PARAM_SSD_TABLE_LIST: str = "__register_ssd_table_list"
# Bool fused param per table to check if the table is offloaded to SSD
FUSED_PARAM_IS_SSD_TABLE: str = "__register_is_ssd_table"


class TBEToRegisterMixIn:
    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        raise NotImplementedError


def get_tbes_to_register_from_iterable(
    iterable: Iterable[torch.nn.Module],
) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
    tbes: Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig] = {}
    for m in iterable:
        if isinstance(m, TBEToRegisterMixIn):
            tbes.update(m.get_tbes_to_register())
    return tbes


def is_fused_param_register_tbe(fused_params: Optional[Dict[str, Any]]) -> bool:
    return (
        # pyrefly: ignore[bad-return]
        fused_params
        and FUSED_PARAM_REGISTER_TBE_BOOL in fused_params
        and fused_params[FUSED_PARAM_REGISTER_TBE_BOOL]
    )


def get_fused_param_tbe_row_alignment(
    fused_params: Optional[Dict[str, Any]],
) -> Optional[int]:
    if fused_params is None or FUSED_PARAM_TBE_ROW_ALIGNMENT not in fused_params:
        return None
    else:
        return fused_params[FUSED_PARAM_TBE_ROW_ALIGNMENT]


def fused_param_bounds_check_mode(
    fused_params: Optional[Dict[str, Any]],
) -> Optional[BoundsCheckMode]:
    if fused_params is None or FUSED_PARAM_BOUNDS_CHECK_MODE not in fused_params:
        return None
    else:
        return fused_params[FUSED_PARAM_BOUNDS_CHECK_MODE]


def fused_param_lengths_to_offsets_lookup(
    fused_params: Optional[Dict[str, Any]],
) -> bool:
    if (
        fused_params is None
        or FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP not in fused_params
    ):
        return False
    else:
        return fused_params[FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP]


def is_fused_param_quant_state_dict_split_scale_bias(
    fused_params: Optional[Dict[str, Any]],
) -> bool:
    return (
        # pyrefly: ignore[bad-return]
        fused_params
        and FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS in fused_params
        and fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS]
    )


def tbe_fused_params(
    fused_params: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not fused_params:
        return None

    fused_params_for_tbe = dict(fused_params)
    if FUSED_PARAM_REGISTER_TBE_BOOL in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_REGISTER_TBE_BOOL)
    if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS)
    if FUSED_PARAM_TBE_ROW_ALIGNMENT in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_TBE_ROW_ALIGNMENT)
    if FUSED_PARAM_BOUNDS_CHECK_MODE in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_BOUNDS_CHECK_MODE)
    if FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP)
    if FUSED_PARAM_SSD_TABLE_LIST in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_SSD_TABLE_LIST)

    return fused_params_for_tbe
