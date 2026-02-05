#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import KVEmbeddingInference
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
    BatchedDenseEmbedding,
    BatchedDenseEmbeddingBag,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.fused_params import (
    fused_param_bounds_check_mode,
    fused_param_lengths_to_offsets_lookup,
    is_fused_param_quant_state_dict_split_scale_bias,
    is_fused_param_register_tbe,
    tbe_fused_params,
    TBEToRegisterMixIn,
)
from torchrec.distributed.types import BoundsCheckMode
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    DATA_TYPE_NUM_BITS,
    data_type_to_sparse_type,
    DataType,
    dtype_to_data_type,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)


def _copy_config(
    original: GroupedEmbeddingConfig,
    data_type: DataType,
    sparse_type: SparseType,
    device: torch.device,
) -> GroupedEmbeddingConfig:
    # Adjust config to quantized version.
    # This obviously doesn't work for column-wise sharding.
    config = copy.deepcopy(original)
    config.data_type = data_type
    for table in config.embedding_tables:
        row_alignment = 16 if device.type == "cuda" else 1
        table.local_cols = rounded_row_size_in_bytes(
            table.local_cols, sparse_type, row_alignment
        )
        if table.local_metadata is not None:
            table.local_metadata.shard_sizes = [
                table.local_rows,
                table.local_cols,
            ]

        global_metadata = table.global_metadata
        if global_metadata is not None:
            for shard_meta in global_metadata.shards_metadata:
                if shard_meta != table.local_metadata:
                    shard_meta.shard_sizes = [
                        shard_meta.shard_sizes[0],
                        rounded_row_size_in_bytes(
                            shard_meta.shard_sizes[1], sparse_type, row_alignment
                        ),
                    ]
            global_metadata.size = torch.Size(
                [
                    global_metadata.size[0],
                    sum(
                        shard_meta.shard_sizes[1]
                        for shard_meta in global_metadata.shards_metadata
                    ),
                ]
            )
    return config


def _quantize_weight(
    state_dict: Dict[str, torch.Tensor],
    data_type: DataType,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    quant_weight_list = []
    for weight in state_dict.values():
        if weight.dtype == torch.float or weight.dtype == torch.float16:
            quantized_weights = (
                torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    weight, DATA_TYPE_NUM_BITS[data_type]
                )
            )
        else:
            raise Exception("Unsupported dtype: {weight.dtype}")

        # weight and 4 byte scale shift (2xfp16)
        quant_weight = quantized_weights[:, :-4]
        scale_shift = quantized_weights[:, -4:]

        quant_weight_list.append((quant_weight, scale_shift))
    return quant_weight_list


def _get_shard_offsets_for_kv_zch(
    config: GroupedEmbeddingConfig,
    shard_index: int,
) -> List[int]:
    """
    Given kv zch tables are rw sharded, getting the row offsets for each shard
    at level to be used witin kv zch look up kernel
    """
    shard_row_offsets = []
    for table in config.embedding_tables:
        assert (
            table.global_metadata is not None
        ), f"Expected global_metadata to be populated for table {table.name} to get shard offsets for kv zch look up kernel"
        assert (
            len(table.global_metadata.shards_metadata) > shard_index
        ), f"Expected table {table.name} to have more shards than shard index {shard_index}. Found {len(table.global_metadata.shards_metadata)} shards"
        shard_row_offsets.append(
            table.global_metadata.shards_metadata[shard_index].shard_offsets[0]
        )
    logger.info(f"Shard row offsets for kv zch look up table: {shard_row_offsets=}")
    return shard_row_offsets


def _get_runtime_device(
    device: Optional[torch.device],
    config: GroupedEmbeddingConfig,
    shard_index: Optional[int] = None,
) -> torch.device:
    index: int = 0 if shard_index is None else shard_index
    if device is not None and device.type != "meta":
        return device
    else:
        return (
            torch.device("cpu")
            if all(
                (
                    table.local_metadata is not None
                    and table.local_metadata.placement is not None
                    and table.local_metadata.placement.device().type == "cpu"
                )
                or (
                    table.global_metadata is not None
                    and len(table.global_metadata.shards_metadata)
                    and table.global_metadata.shards_metadata[index].placement
                    is not None
                    and table.global_metadata.shards_metadata[index]
                    .placement.device()
                    .type
                    == "cpu"
                )
                for table in config.embedding_tables
            )
            else torch.device("cuda")
        )


@torch.fx.wrap
def _unwrap_kjt(
    features: KeyedJaggedTensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Here it should always follow cuda path, runtime device cannot be meta
    indices = features.values()
    offsets = features.offsets()
    return (
        indices.int(),  # currently only support int32 indices
        offsets.int(),
        features.weights_or_none(),
    )


def _unwrap_kjt_for_cpu(
    features: KeyedJaggedTensor, weighted: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    indices = features.values()
    offsets = features.offsets().type(indices.dtype)
    if weighted:
        return indices, offsets, features.weights()
    else:
        return indices, offsets, None


@torch.fx.wrap
def _unwrap_kjt_lengths(
    features: KeyedJaggedTensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    indices = features.values()
    lengths = features.lengths()
    return (
        indices.int(),
        lengths.int(),
        features.weights_or_none(),
    )


@torch.fx.wrap
def _unwrap_optional_tensor(
    tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    # Typing for TorchScript
    assert tensor is not None
    return tensor


class IntNBitTableBatchedEmbeddingBagsCodegenWithLength(
    IntNBitTableBatchedEmbeddingBagsCodegen
):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            indices=indices,
            offsets=(torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)),
            per_sample_weights=per_sample_weights,
        )


class QuantBatchedEmbeddingBag(
    BaseBatchedEmbeddingBag[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ],
    TBEToRegisterMixIn,
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        shard_index: Optional[int] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        is_virtual_table: bool = False
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                managed.append(EmbeddingLocation.HOST)
            if table.use_virtual_table:
                is_virtual_table = True
        self._config: GroupedEmbeddingConfig = config
        self._emb_module_registered: bool = is_fused_param_register_tbe(fused_params)
        self._is_weighted: Optional[bool] = config.is_weighted
        self._quant_state_dict_split_scale_bias: bool = (
            is_fused_param_quant_state_dict_split_scale_bias(fused_params)
        )
        bounds_check_mode: Optional[BoundsCheckMode] = fused_param_bounds_check_mode(
            fused_params
        )

        self._runtime_device: torch.device = _get_runtime_device(
            device, config, shard_index
        )
        # 16 for CUDA, 1 for others like CPU and MTIA.
        self._tbe_row_alignment: int = 16 if self._runtime_device.type == "cuda" else 1
        embedding_specs = []
        for local_rows, local_cols, table, location in zip(
            self._local_rows,
            self._local_cols,
            config.embedding_tables,
            managed,
        ):
            embedding_specs.append(
                (
                    table.name,
                    local_rows,
                    (
                        local_cols
                        if self._quant_state_dict_split_scale_bias
                        else table.embedding_dim
                    ),
                    data_type_to_sparse_type(table.data_type),
                    location,
                )
            )

        self.lengths_to_tbe: bool = fused_param_lengths_to_offsets_lookup(fused_params)

        if self.lengths_to_tbe:
            tbe_clazz = IntNBitTableBatchedEmbeddingBagsCodegenWithLength
        elif is_virtual_table:
            tbe_clazz = KVEmbeddingInference
        else:
            tbe_clazz = IntNBitTableBatchedEmbeddingBagsCodegen

        if is_virtual_table:
            assert (
                shard_index is not None and shard_index >= 0
            ), "valid shard_index must be provided for kv zch batch embedding to compute shard offsets"
            shard_offsets_for_kv_zch = _get_shard_offsets_for_kv_zch(
                config, shard_index
            )
        else:
            shard_offsets_for_kv_zch = None

        # Determine embedding cache mode for KV embedding tables
        embedding_cache_mode = False  # Default: False = randomized initialization
        if tbe_clazz == KVEmbeddingInference:
            # For KV embedding tables, set cache mode based on embedding table configuration
            # Check if any table has NoEvictionPolicy - use zero init for those
            for table in config.embedding_tables:
                if (
                    table.virtual_table_eviction_policy is not None
                    and type(table.virtual_table_eviction_policy).__name__
                    == "NoEvictionPolicy"
                ):
                    embedding_cache_mode = True  # True = zero initialization
                    break

        # Build kwargs for module construction
        module_kwargs: Dict[str, Any] = {
            "embedding_specs": embedding_specs,
            "device": device,
            "pooling_mode": self._pooling,
            "feature_table_map": self._feature_table_map,
            "row_alignment": self._tbe_row_alignment,
            "uvm_host_mapped": True,  # Use cudaHostAlloc for UVM CACHING to fix imbalance numa memory issue
            "bounds_check_mode": (
                bounds_check_mode if bounds_check_mode else BoundsCheckMode.WARNING
            ),
            "feature_names_per_table": [
                table.feature_names for table in config.embedding_tables
            ],
        }

        # Add KV-specific parameters
        if tbe_clazz == KVEmbeddingInference:
            module_kwargs["embedding_cache_mode"] = embedding_cache_mode

        # Add fused params
        module_kwargs.update(**(tbe_fused_params(fused_params) or {}))

        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = tbe_clazz(
            **module_kwargs
        )
        if device is not None:
            self._emb_module.initialize_weights()
        if shard_offsets_for_kv_zch is not None:
            assert (
                tbe_clazz == KVEmbeddingInference
            ), "shard_offsets_for_kv_zch should be computed only for kv zch kernel"
            self._emb_module.init_tbe_config(shard_offsets_for_kv_zch)

    def init_parameters(self) -> None:
        pass

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return {self._emb_module: self._config}

    def _emb_module_forward(
        self,
        indices: torch.Tensor,
        lengths_or_offsets: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        kwargs = {"indices": indices}

        if self.lengths_to_tbe:
            kwargs["lengths"] = lengths_or_offsets
        else:
            kwargs["offsets"] = lengths_or_offsets

        if self._is_weighted:
            kwargs["per_sample_weights"] = _unwrap_optional_tensor(weights)

        if self._emb_module_registered:
            # Conditional call of .forward function for FX:
            # emb_module() can go through FX only if emb_module is registered in named_modules (FX node call_module)
            # emb_module.forward() does not require registering emb_module in named_modules (FX node call_function)
            # For some post processing that requires TBE emb_module copied in fx.GraphModule we need to be call_module, as it will copies this module inside fx.GraphModule unchanged.
            return self._emb_module(**kwargs)
        else:
            return self._emb_module.forward(**kwargs)

    # TODO: T232803649 Remove unused VBE parameters from the forward signature
    def forward(
        self,
        features: KeyedJaggedTensor,
        vbe_output: Optional[torch.Tensor] = None,  # unused
        vbe_output_offsets: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        # Important: _unwrap_kjt regex for FX tracing TAGing
        lengths, offsets = None, None
        if self._runtime_device.type == "cpu":
            if self.lengths_to_tbe:
                indices, lengths, per_sample_weights = _unwrap_kjt_lengths(features)
            else:
                indices, offsets, per_sample_weights = _unwrap_kjt_for_cpu(
                    features, self._config.is_weighted
                )
        else:
            if self.lengths_to_tbe:
                indices, lengths, per_sample_weights = _unwrap_kjt_lengths(features)
            else:
                indices, offsets, per_sample_weights = _unwrap_kjt(features)

        return self._emb_module_forward(
            indices, lengths if lengths is not None else offsets, per_sample_weights
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in QuantBatchedEmbeddingBag.named_split_embedding_weights"
        for config, (weight, weight_qscale, weight_qbias) in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            ),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight
            if self._quant_state_dict_split_scale_bias:
                #  `Generator[Tuple[str, Optional[Tensor]], Any, Any]`.
                yield append_prefix(
                    prefix, f"{config.name}.weight_qscale"
                ), weight_qscale
                #  `Generator[Tuple[str, Optional[Tensor]], Any, Any]`.
                yield append_prefix(prefix, f"{config.name}.weight_qbias"), weight_qbias

    def split_embedding_weights(
        self,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        return [
            (weight, qscale, qbias)
            for weight, qscale, qbias in self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            )
        ]

    @classmethod
    def from_float(
        cls, module: BaseEmbedding, use_precomputed_fake_quant: bool = False
    ) -> "QuantBatchedEmbeddingBag":
        assert hasattr(
            module, "qconfig"
        ), "BaseEmbedding input float module must have qconfig defined"

        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        sparse_type = data_type_to_sparse_type(data_type)

        # TODO Can we simplify this with state_dict = module.state_dict()?
        state_dict = (
            dict(module.named_split_embedding_weights())
            if isinstance(module, BatchedDenseEmbeddingBag)
            else dict(module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        config = _copy_config(module.config, data_type, sparse_type, device)
        ret = QuantBatchedEmbeddingBag(config=config, device=device)

        quant_weight_list = _quantize_weight(state_dict, data_type)
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret


class QuantBatchedEmbedding(
    BaseBatchedEmbedding[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ],
    TBEToRegisterMixIn,
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        shard_index: Optional[int] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        is_virtual_table = False
        embedding_cache_mode = False
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                managed.append(EmbeddingLocation.HOST)
            if table.use_virtual_table:
                is_virtual_table = True
            if table.enable_embedding_update:
                embedding_cache_mode = True
        self._config: GroupedEmbeddingConfig = config
        self._emb_module_registered: bool = is_fused_param_register_tbe(fused_params)
        self._quant_state_dict_split_scale_bias: bool = (
            is_fused_param_quant_state_dict_split_scale_bias(fused_params)
        )
        self._runtime_device: torch.device = _get_runtime_device(
            device, config, shard_index
        )
        # 16 for CUDA, 1 for others like CPU and MTIA.
        self._tbe_row_alignment: int = 16 if self._runtime_device.type == "cuda" else 1
        embedding_clazz = (
            KVEmbeddingInference
            if is_virtual_table
            else IntNBitTableBatchedEmbeddingBagsCodegen
        )
        if is_virtual_table:
            assert (
                shard_index is not None and shard_index >= 0
            ), "valid shard_index must be provided for kv zch batch embedding to compute shard offsets"
            shard_offsets_for_kv_zch = _get_shard_offsets_for_kv_zch(
                config, shard_index
            )
        else:
            shard_offsets_for_kv_zch = None

        # Build kwargs for module construction
        module_kwargs: Dict[str, Any] = {
            "embedding_specs": [
                (
                    table.name,
                    local_rows,
                    (
                        local_cols
                        if self._quant_state_dict_split_scale_bias
                        else table.embedding_dim
                    ),
                    data_type_to_sparse_type(table.data_type),
                    location,
                )
                for local_rows, local_cols, table, location in zip(
                    self._local_rows,
                    self._local_cols,
                    config.embedding_tables,
                    managed,
                )
            ],
            "device": device,
            "pooling_mode": PoolingMode.NONE,
            "feature_table_map": self._feature_table_map,
            "row_alignment": self._tbe_row_alignment,
            "uvm_host_mapped": True,  # Use cudaHostAlloc for UVM CACHING to fix imbalance numa memory issue
            "feature_names_per_table": [
                table.feature_names for table in config.embedding_tables
            ],
        }

        # Add KV-specific parameters
        if embedding_clazz == KVEmbeddingInference:
            module_kwargs["embedding_cache_mode"] = embedding_cache_mode

        # Add fused params
        module_kwargs.update(**(tbe_fused_params(fused_params) or {}))

        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = embedding_clazz(
            **module_kwargs
        )
        if device is not None:
            self._emb_module.initialize_weights()
        if shard_offsets_for_kv_zch is not None:
            assert (
                embedding_clazz == KVEmbeddingInference
            ), "shard_offsets_for_kv_zch should be computed only for kv zch kernel"
            self._emb_module.init_tbe_config(shard_offsets_for_kv_zch)

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return {self._emb_module: self._config}

    def split_embedding_weights(
        self,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        return [
            (weight, qscale, qbias)
            for weight, qscale, qbias in self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            )
        ]

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        if self._runtime_device.type == "cpu":
            # To distinguish fx tracing on CPU embedding.
            values, offsets, _ = _unwrap_kjt_for_cpu(
                features, weighted=self._config.is_weighted
            )
        else:
            values, offsets, _ = _unwrap_kjt(features)

        if self._emb_module_registered:
            return self.emb_module(
                indices=values,
                offsets=offsets,
            )
        else:
            return self.emb_module.forward(
                indices=values,
                offsets=offsets,
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for config, (weight, weight_qscale, weight_qbias) in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            ),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight
            if self._quant_state_dict_split_scale_bias:
                #  `Generator[Tuple[str, Optional[Tensor]], Any, Any]`.
                yield append_prefix(
                    prefix, f"{config.name}.weight_qscale"
                ), weight_qscale
                #  `Generator[Tuple[str, Optional[Tensor]], Any, Any]`.
                yield append_prefix(prefix, f"{config.name}.weight_qbias"), weight_qbias

    @classmethod
    def from_float(
        cls, module: BaseEmbedding, use_precomputed_fake_quant: bool = False
    ) -> "QuantBatchedEmbedding":
        assert hasattr(
            module, "qconfig"
        ), "BaseEmbedding input float module must have qconfig defined"

        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        sparse_type = data_type_to_sparse_type(data_type)

        # TODO Can we simplify this with state_dict = module.state_dict()?
        state_dict = (
            dict(module.named_split_embedding_weights())
            if isinstance(module, BatchedDenseEmbedding)
            else dict(module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        config = _copy_config(module.config, data_type, sparse_type, device)
        ret = QuantBatchedEmbedding(config=config, device=device)

        quant_weight_list = _quantize_weight(state_dict, data_type)
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret
