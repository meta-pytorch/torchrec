#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from abc import ABC
from collections import OrderedDict
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BackendType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.ssd.training import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.distributed._tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
    BatchedDenseEmbedding,
    BatchedDenseEmbeddingBag,
    BatchedFusedEmbedding,
    BatchedFusedEmbeddingBag,
    KeyValueEmbedding,
    KeyValueEmbeddingBag,
    ShardedBatchedFusedEmbedding,
    ShardedBatchedFusedEmbeddingBag,
    SparseType,
    TritonBatchedFusedEmbeddingBag,
    ZeroCollisionEmbeddingCache,
    ZeroCollisionKeyValueEmbedding,
    ZeroCollisionKeyValueEmbeddingBag,
)
from torchrec.distributed.comm_ops import get_gradient_division
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_types import (
    BaseEmbeddingLookup,
    BaseEmbeddingUpdate,
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    InputDistOutputs,
)
from torchrec.distributed.fused_params import (
    get_tbes_to_register_from_iterable,
    TBEToRegisterMixIn,
)
from torchrec.distributed.global_settings import get_propogate_device
from torchrec.distributed.quant_embedding_kernel import (
    QuantBatchedEmbedding,
    QuantBatchedEmbeddingBag,
)
from torchrec.distributed.types import (
    LazyAwaitable,
    rank_device,
    ShardedTensor,
    ShardingEnv,
    ShardingEnv2D,
    ShardingStrategy,
    ShardingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


@torch.fx.wrap
def fx_wrap_tensor_view2d(x: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return x.view(dim0, dim1)


@torch.fx.wrap
def dummy_tensor(
    sparse_features: KeyedJaggedTensor, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty([0], dtype=dtype, device=sparse_features.device()).view(
        sparse_features.stride(), 0
    )


def _load_state_dict(
    emb_modules: "nn.ModuleList",
    state_dict: "OrderedDict[str, Union[torch.Tensor, ShardedTensor, DTensor]]",
) -> Tuple[List[str], List[str]]:
    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    for emb_module in emb_modules:
        for key, dst_param in emb_module.state_dict().items():
            if key in state_dict:
                src_param = state_dict[key]
                if isinstance(dst_param, ShardedTensor):
                    assert isinstance(src_param, ShardedTensor)
                    assert len(dst_param.local_shards()) == len(
                        src_param.local_shards()
                    )
                    for dst_local_shard, src_local_shard in zip(
                        dst_param.local_shards(), src_param.local_shards()
                    ):
                        assert (
                            dst_local_shard.metadata.shard_offsets
                            == src_local_shard.metadata.shard_offsets
                        )
                        assert (
                            dst_local_shard.metadata.shard_sizes
                            == src_local_shard.metadata.shard_sizes
                        )

                        dst_local_shard.tensor.detach().copy_(src_local_shard.tensor)
                elif isinstance(dst_param, DTensor):
                    assert isinstance(src_param, DTensor)
                    assert len(dst_param.to_local().local_chunks) == len(
                        src_param.to_local().local_chunks
                    )
                    for i, (dst_local_shard, src_local_shard) in enumerate(
                        zip(
                            dst_param.to_local().local_shards(),
                            src_param.to_local().local_shards(),
                        )
                    ):
                        assert (
                            dst_param.to_local().local_chunks[i]
                            == src_param.to_local().local_chunks[i]
                        )
                        dst_local_shard.detach().copy_(src_local_shard)
                else:
                    assert isinstance(src_param, torch.Tensor) and isinstance(
                        dst_param, torch.Tensor
                    )
                    dst_param.detach().copy_(src_param)
                unexpected_keys.remove(key)
            else:
                missing_keys.append(key)
    return missing_keys, unexpected_keys


@torch.fx.wrap
def embeddings_cat_empty_rank_handle(
    embeddings: List[torch.Tensor],
    dummy_embs_tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    if len(embeddings) == 0:
        # a hack for empty ranks
        return dummy_embs_tensor
    elif len(embeddings) == 1:
        return embeddings[0]
    else:
        return torch.cat(embeddings, dim=dim)


@torch.fx.wrap
def embeddings_cat_empty_rank_handle_inference(
    embeddings: List[torch.Tensor],
    dim: int = 0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if len(embeddings) == 0:
        # return a dummy empty tensor when grouped_configs is empty
        dev: Optional[torch.device] = (
            torch.device(device) if device is not None else None
        )
        return torch.empty([0], dtype=dtype, device=dev)
    elif len(embeddings) == 1:
        return embeddings[0]
    else:
        return torch.cat(embeddings, dim=dim)


class GroupedEmbeddingsLookup(BaseEmbeddingLookup[KeyedJaggedTensor, torch.Tensor]):
    """
    Lookup modules for Sequence embeddings (i.e Embeddings)
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        env: Optional[ShardingEnv] = None,
    ) -> None:
        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        self._need_prefetch: bool = False
        for config in grouped_configs:
            self._emb_modules.append(
                self._create_embedding_kernel(config, pg, device, env)
            )

        self._feature_splits: List[int] = []
        for config in grouped_configs:
            self._feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs
        # Model tracker function to tracker optimizer state
        self.optim_state_tracker_fn: Optional[
            Callable[[KeyedJaggedTensor, torch.Tensor, Optional[nn.Module]], None]
        ] = None

    def _create_embedding_kernel(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        env: Optional[ShardingEnv] = None,
    ) -> BaseEmbedding:
        for table in config.embedding_tables:
            if (
                table.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING
                or table.compute_kernel == EmbeddingComputeKernel.KEY_VALUE
                or table.compute_kernel == EmbeddingComputeKernel.SSD_VIRTUAL_TABLE
                or table.compute_kernel == EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE
            ):
                self._need_prefetch = True
        if config.compute_kernel == EmbeddingComputeKernel.DENSE:
            return BatchedDenseEmbedding(
                config=config,
                pg=pg,
                device=device,
            )
        elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
            if (
                env
                and isinstance(env, ShardingEnv2D)
                and env.sharding_strategy == ShardingStrategy.FULLY_SHARDED
            ):
                return ShardedBatchedFusedEmbedding(
                    config=config, pg=pg, device=device, env=env
                )
            else:
                return BatchedFusedEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
        elif config.compute_kernel == EmbeddingComputeKernel.KEY_VALUE:
            return KeyValueEmbedding(
                config=config,
                pg=pg,
                device=device,
            )
        elif config.compute_kernel == EmbeddingComputeKernel.SSD_VIRTUAL_TABLE:
            # for ssd kv
            return ZeroCollisionKeyValueEmbedding(
                config=config,
                pg=pg,
                device=device,
                backend_type=BackendType.SSD,
            )
        elif config.compute_kernel == EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE:
            # for dram kv
            if config.enable_embedding_update:
                return ZeroCollisionEmbeddingCache(
                    config=config,
                    pg=pg,
                    device=device,
                    backend_type=BackendType.DRAM,
                )
            else:
                return ZeroCollisionKeyValueEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                    backend_type=BackendType.DRAM,
                )
        else:
            raise ValueError(f"Compute kernel not supported {config.compute_kernel}")

    def prefetch(
        self,
        sparse_features: KeyedJaggedTensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if not self._need_prefetch:
            return
        if len(self._emb_modules) > 0:
            assert sparse_features is not None
            features_by_group = sparse_features.split(
                self._feature_splits,
            )
            for emb_op, features in zip(self._emb_modules, features_by_group):
                if (
                    isinstance(
                        emb_op.emb_module,
                        (
                            SplitTableBatchedEmbeddingBagsCodegen,
                            SSDTableBatchedEmbeddingBags,
                        ),
                    )
                    #  attribute `prefetch_pipeline`.
                    and not emb_op.emb_module.prefetch_pipeline
                ):
                    logging.error(
                        f"Invalid setting on {type(emb_op.emb_module)} modules. prefetch_pipeline must be set to True.\n"
                        "If you don’t turn on prefetch_pipeline, cache locations might be wrong in backward and can cause wrong results.\n"
                    )
                if hasattr(emb_op.emb_module, "prefetch"):
                    #  attribute `prefetch`.
                    emb_op.emb_module.prefetch(
                        indices=features.values(),
                        offsets=features.offsets(),
                        forward_stream=forward_stream,
                    )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        features_by_group = sparse_features.split(
            self._feature_splits,
        )
        for emb_op, features in zip(self._emb_modules, features_by_group):
            lookup = emb_op(features).view(-1)
            embeddings.append(lookup)

            # Model tracker optimizer state function, will only be set called
            # when model tracker is configured to track optimizer state
            if self.optim_state_tracker_fn is not None:
                self.optim_state_tracker_fn(features, lookup, emb_op)

        return embeddings_cat_empty_rank_handle(embeddings, self._dummy_embs_tensor)

    def get_resize_awaitables(self) -> List[LazyAwaitable[torch.Tensor]]:
        # TODO - we can probably do some smart grouping to make this more efficient
        return [
            emb_module.get_rs_awaitable()
            for emb_module in self._emb_modules
            if hasattr(emb_module, "get_rs_awaitable")
        ]

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[torch.Tensor, ShardedTensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "GroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "GroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for embedding_kernel in self._emb_modules:
            for (
                table_name,
                tbe_slice,
            ) in embedding_kernel.named_parameters_by_table():
                yield (table_name, tbe_slice)

    def get_named_split_embedding_weights_snapshot(
        self,
    ) -> Iterator[
        Tuple[
            str,
            Union[ShardedTensor, PartiallyMaterializedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
        ]
    ]:
        """
        Return an iterator over embedding tables, yielding both the table name as well as the embedding
        table itself. The embedding table is in the form of PartiallyMaterializedTensor with a valid
        RocksDB snapshot to support windowed access.
        """
        for emb_module in self._emb_modules:
            if isinstance(emb_module, KeyValueEmbedding) or isinstance(
                emb_module, ZeroCollisionKeyValueEmbedding
            ):
                yield from emb_module.get_named_split_embedding_weights_snapshot()

    def flush(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.flush()

    def create_rocksdb_hard_link_snapshot(self) -> None:
        for emb_module in self._emb_modules:
            if isinstance(emb_module, KeyValueEmbedding):
                emb_module.create_rocksdb_hard_link_snapshot()

    def purge(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.purge()

    def register_optim_state_tracker_fn(
        self,
        record_fn: Callable[
            [KeyedJaggedTensor, torch.Tensor, Optional[nn.Module]], None
        ],
    ) -> None:
        """
        Model tracker function to tracker optimizer state

         Args:
             record_fn (Callable[[KeyedJaggedTensor, torch.Tensor, Optional[nn.Module]], None]): A custom record function to be called after lookup is done.

        """
        self.optim_state_tracker_fn = record_fn


class GroupedEmbeddingsUpdate(BaseEmbeddingUpdate[KeyedJaggedTensor]):
    """
    Update modules for Sequence embeddings (i.e Embeddings)
    """

    def __init__(
        self,
        grouped_embeddings_lookup: GroupedEmbeddingsLookup,
    ) -> None:
        super().__init__()
        self._emb_modules: List[BaseEmbedding] = []
        self._feature_splits: List[int] = []
        for emb_module in grouped_embeddings_lookup._emb_modules:
            emb_module = cast(BaseBatchedEmbedding[torch.Tensor], emb_module)
            if emb_module.config.enable_embedding_update:
                self._feature_splits.append(emb_module.config.num_features())
                self._emb_modules.append(emb_module)

    def forward(self, embeddings: KeyedJaggedTensor) -> None:
        features_by_group = embeddings.split(
            self._feature_splits,
        )
        for emb_module, features in zip(self._emb_modules, features_by_group):
            emb_module.update(features)


class CommOpGradientScaling(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx, input_tensor: torch.Tensor, scale_gradient_factor: int
    ) -> torch.Tensor:
        ctx.scale_gradient_factor = scale_gradient_factor
        return input_tensor

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        # When gradient division is on, we scale down the gradient by world size
        # at alltoall backward for model parallelism. However weights
        # is controlled by DDP so it already has gradient division, so we scale
        # the gradient back up
        grad_output.mul_(ctx.scale_gradient_factor)
        return grad_output, None


class GroupedPooledEmbeddingsLookup(
    BaseEmbeddingLookup[KeyedJaggedTensor, torch.Tensor]
):
    """
    Lookup modules for Pooled embeddings (i.e EmbeddingBags)
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        pg: Optional[dist.ProcessGroup] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
        scale_weight_gradients: bool = True,
        sharding_type: Optional[ShardingType] = None,
        env: Optional[ShardingEnv] = None,
    ) -> None:
        super().__init__()
        self._env = env
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(
                self._create_embedding_kernel(config, device, pg, sharding_type, env)
            )

        self._feature_splits: List[int] = []
        for config in grouped_configs:
            self._feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs
        self._feature_processor = feature_processor

        self._world_size: int = dist.get_world_size(pg)
        self._scale_gradient_factor: int = (
            self._world_size
            if scale_weight_gradients and get_gradient_division()
            else 1
        )
        # Model tracker function to tracker optimizer state
        self.optim_state_tracker_fn: Optional[
            Callable[[KeyedJaggedTensor, torch.Tensor, Optional[nn.Module]], None]
        ] = None

    def _create_embedding_kernel(
        self,
        config: GroupedEmbeddingConfig,
        device: Optional[torch.device],
        pg: Optional[dist.ProcessGroup],
        sharding_type: Optional[ShardingType],
        env: Optional[ShardingEnv],
    ) -> BaseEmbedding:
        if config.compute_kernel == EmbeddingComputeKernel.DENSE:
            return BatchedDenseEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
                sharding_type=sharding_type,
            )
        elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
            if (
                env
                and isinstance(env, ShardingEnv2D)
                and env.sharding_strategy == ShardingStrategy.FULLY_SHARDED
            ):
                return ShardedBatchedFusedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                    env=env,
                )
            else:
                return BatchedFusedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    sharding_type=sharding_type,
                    env=env,
                )
        elif config.compute_kernel == EmbeddingComputeKernel.FUSED_TRITON:
            return TritonBatchedFusedEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
            )
        elif config.compute_kernel in {
            EmbeddingComputeKernel.KEY_VALUE,
        }:
            return KeyValueEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
                sharding_type=sharding_type,
            )
        elif config.compute_kernel == EmbeddingComputeKernel.SSD_VIRTUAL_TABLE:
            # for ssd kv
            return ZeroCollisionKeyValueEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
                sharding_type=sharding_type,
                backend_type=BackendType.SSD,
            )
        elif config.compute_kernel == EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE:
            # for dram kv
            return ZeroCollisionKeyValueEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
                sharding_type=sharding_type,
                backend_type=BackendType.DRAM,
            )
        else:
            raise ValueError(f"Compute kernel not supported {config.compute_kernel}")

    def prefetch(
        self,
        sparse_features: KeyedJaggedTensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        def _need_prefetch(config: GroupedEmbeddingConfig) -> bool:
            for table in config.embedding_tables:
                if (
                    table.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING
                    or table.compute_kernel == EmbeddingComputeKernel.KEY_VALUE
                    or table.compute_kernel == EmbeddingComputeKernel.SSD_VIRTUAL_TABLE
                    or table.compute_kernel == EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE
                ):
                    return True
            return False

        if len(self._emb_modules) > 0:
            assert sparse_features is not None
            features_by_group = sparse_features.split(
                self._feature_splits,
            )
            for emb_op, features in zip(self._emb_modules, features_by_group):
                #  but got `Union[Module, Tensor]`.
                if not _need_prefetch(emb_op.config):
                    continue
                if (
                    isinstance(
                        emb_op.emb_module,
                        (
                            SplitTableBatchedEmbeddingBagsCodegen,
                            SSDTableBatchedEmbeddingBags,
                        ),
                    )
                    #  attribute `prefetch_pipeline`.
                    and not emb_op.emb_module.prefetch_pipeline
                ):
                    logging.error(
                        f"Invalid setting on {type(emb_op.emb_module)} modules. prefetch_pipeline must be set to True.\n"
                        "If you don't turn on prefetch_pipeline, cache locations might be wrong in backward and can cause wrong results.\n"
                    )
                if hasattr(emb_op.emb_module, "prefetch"):
                    #  attribute `prefetch`.
                    emb_op.emb_module.prefetch(
                        indices=features.values(),
                        offsets=features.offsets(),
                        forward_stream=forward_stream,
                        batch_size_per_feature_per_rank=(
                            features.stride_per_key_per_rank()
                            if features.variable_stride_per_key()
                            else None
                        ),
                    )

    def _merge_variable_batch_embeddings(
        self, embeddings: List[torch.Tensor], splits: List[List[int]]
    ) -> torch.Tensor:
        assert len(embeddings) > 1 and len(splits) > 1

        logger.info(
            "[Deprecated] Merge VBE embeddings from the following TBEs "
            f"(world size: {self._world_size}):\n"
            + "\n".join(
                [
                    f"\t{module.__class__.__name__}:{len(split)} splits"
                    for module, split in zip(self._emb_modules, splits)
                ]
            )
        )

        split_embs = [e.split(s) for e, s in zip(embeddings, splits)]
        combined_embs = [
            emb
            for rank in range(self._world_size)
            for n, embs in zip(self._feature_splits, split_embs)
            for emb in embs[n * rank : n * rank + n]
        ]
        return torch.cat(combined_embs)

    def _vbe_splits(
        self, features_by_group: List[KeyedJaggedTensor]
    ) -> List[List[int]]:
        """Calculates the split of features for each TBE when VBE is enabled.

        Example:
            # Returned splits for 2 TBEs, 3 features, world size of 2
            [
                # TBE 0: 2 features
                [
                    2,    # Feature 0 @rank 0
                    3,    # Feature 1 @rank 0
                    1,    # Feature 0 @rank 1
                    2,    # Feature 1 @rank 1
                ],
                # TBE 1: 1 feature
                [
                    4,    # Feature 2 @rank 0
                    2,    # Feature 2 @rank 1
                ],
            ]
        """
        splits = []
        for config, features in zip(self.grouped_configs, features_by_group):
            embedding_dim_per_key = config.embedding_dims()
            stride_per_rank_per_key = list(zip(*features.stride_per_key_per_rank()))
            splits.append(
                [
                    stride * dim
                    for stride_per_key in stride_per_rank_per_key
                    for stride, dim in zip(stride_per_key, embedding_dim_per_key)
                ]
            )
        return splits

    def _vbe_output_dtype(self) -> torch.dtype:
        fused_params = self.grouped_configs[0].fused_params or {}
        first_dtype: SparseType = fused_params.get("output_dtype", SparseType.FP32)

        for config in self.grouped_configs:
            fused_params = config.fused_params or {}
            dtype = fused_params.get("output_dtype", SparseType.FP32)
            assert (
                dtype == first_dtype
            ), f"Mismatch output_dtype: {dtype} != {first_dtype}"

        return torch.float32 if first_dtype == SparseType.FP32 else torch.float16

    def _create_vbe_output_and_offsets(
        self, vbe_splits: List[List[int]], device: torch.device
    ) -> Tuple[torch.Tensor, List[List[List[int]]]]:
        """Creates the output tensor and offsets for TBEs with VBE enabled.

        The output tensor is a pre-allocated empty 1D tensor to store the lookup
        results from all TBEs. The offsets are used by the TBEs to correctly
        populate the output tensor with the lookup results.

        Example:
            # Given the following splits for 2 TBEs and a world size of 2:
            [
                # TBE 0: 2 features
                [
                    2,    # Feature 0 @rank 0
                    3,    # Feature 1 @rank 0
                    1,    # Feature 0 @rank 1
                    2,    # Feature 1 @rank 1
                ],
                # TBE 1: 1 feature
                [
                    4,    # Feature 2 @rank 0
                    2,    # Feature 2 @rank 1
                ],
            ]

            # Output tensor should be of size 14
            # Returned VBE offsets:
            [
                # TBE 0: 2 features
                [
                    [0, 2],  # [Feature 0, Feature 1] @rank 0
                    [9, 10], # [Feature 0, Feature 1] @rank 1
                ],
                # TBE 1: 1 feature
                [
                    [5],     # [Feature 2] @rank 0
                    [12],    # [Feature 2] @rank 1
                ],
            ]
        """
        # Pre-allocates 1D output tensor to avoid expensive merging
        vbe_output = torch.empty(
            sum([sum(split) for split in vbe_splits]),
            dtype=self._vbe_output_dtype(),
        ).to(device)

        # Calculates the offsets for features of each TBE when placed in
        # the 1D pre-allocated output tensor above, ordered by rank.
        vbe_offsets = [[] for _ in range(len(self._emb_modules))]
        current_offset = 0
        for rank in range(self._world_size):
            for n, offsets, split in zip(self._feature_splits, vbe_offsets, vbe_splits):
                split_offsets = []
                for size in split[n * rank : n * rank + n]:
                    split_offsets.append(current_offset)
                    current_offset += size

                offsets.append(split_offsets)

        return vbe_output, vbe_offsets

    def _apply_fearture_processor_and_gradient_scaling(
        self,
        features: KeyedJaggedTensor,
        config: GroupedEmbeddingConfig,
    ) -> KeyedJaggedTensor:
        if (
            config.has_feature_processor
            and self._feature_processor is not None
            and isinstance(self._feature_processor, BaseGroupedFeatureProcessor)
        ):
            features = self._feature_processor(features)

        if config.is_weighted:
            features._weights = CommOpGradientScaling.apply(
                features._weights, self._scale_gradient_factor
            )
        return features

    def _forward(
        self,
        features_by_group: List[KeyedJaggedTensor],
    ) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []

        for config, emb_op, features in zip(
            self.grouped_configs,
            self._emb_modules,
            features_by_group,
        ):
            features = self._apply_fearture_processor_and_gradient_scaling(
                features, config
            )
            lookup = emb_op(features)
            embeddings.append(lookup)

            # Model tracker optimizer state function, will only be set called
            # when model tracker is configured to track optimizer state
            if self.optim_state_tracker_fn is not None:
                self.optim_state_tracker_fn(features, lookup, emb_op)

        return embeddings

    def _forward_with_vbe_merging(
        self,
        features_by_group: List[KeyedJaggedTensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Merges the TBE output when VBE is enabled and multiple TBEs are
        involved.

        If opt-in for the preallocated merge approach, an 1D empty tensor will be
        preallocated for TBEs to handle the merging logic. Otherwise, the output
        will be split and then merged via `torch.cat`.
        """
        vbe_splits = self._vbe_splits(features_by_group)

        # If we do not opt-in to pre-allocate the VBE output, we need to run
        # forward for each TBE individually and merge the results via
        # `torch.cat`.
        if not torch._utils_internal.justknobs_check(
            "pytorch/torchrec:killswitch_enable_preallocated_vbe_merge"
        ):
            return self._merge_variable_batch_embeddings(
                self._forward(features_by_group), vbe_splits
            )

        vbe_output, vbe_offsets = self._create_vbe_output_and_offsets(
            vbe_splits, device
        )

        vbe_offsets = [torch.tensor(offsets, device=device) for offsets in vbe_offsets]

        for config, emb_op, features, offsets in zip(
            self.grouped_configs,
            self._emb_modules,
            features_by_group,
            vbe_offsets,
        ):
            features = self._apply_fearture_processor_and_gradient_scaling(
                features, config
            )

            # With a pre-allocated output tensor, each TBE uses the output from
            # the previous TBE as input, allowing autograd to correctly track
            # the backward pass. TBEs should use `vbe_output_offsets` to update
            # the preallocated tensor partially.
            vbe_output = emb_op(
                features,
                vbe_output_offsets=offsets,
                vbe_output=vbe_output,
            )

            # Model tracker optimizer state function, will only be set called
            # when model tracker is configured to track optimizer state
            if self.optim_state_tracker_fn is not None:
                self.optim_state_tracker_fn(features, vbe_output, emb_op)

        return vbe_output

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        is_vbe_enabled = sparse_features.variable_stride_per_key()

        dummy_embedding = (
            self._dummy_embs_tensor
            if is_vbe_enabled
            else fx_wrap_tensor_view2d(
                self._dummy_embs_tensor, sparse_features.stride(), 0
            )
        )

        if len(self._emb_modules) == 0:
            return dummy_embedding

        features_by_group = sparse_features.split(self._feature_splits)

        # If VBE is enabled and mulitple TBEs are involved, we need to merge the
        # output
        if is_vbe_enabled and len(self._emb_modules) > 1:
            return self._forward_with_vbe_merging(
                features_by_group, device=sparse_features.device()
            )

        embeddings = self._forward(features_by_group)
        return embeddings_cat_empty_rank_handle(
            embeddings,
            dummy_embedding,
            dim=1,
        )

    def get_resize_awaitables(self) -> List[LazyAwaitable[torch.Tensor]]:
        # TODO - we can probably do some smart grouping to make this more efficient
        return [
            emb_module.get_rs_awaitable()
            for emb_module in self._emb_modules
            if hasattr(emb_module, "get_rs_awaitable")
        ]

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "GroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "GroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for embedding_kernel in self._emb_modules:
            for (
                table_name,
                tbe_slice,
            ) in embedding_kernel.named_parameters_by_table():
                yield (table_name, tbe_slice)

    def get_named_split_embedding_weights_snapshot(
        self,
    ) -> Iterator[
        Tuple[
            str,
            Union[ShardedTensor, PartiallyMaterializedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
        ]
    ]:
        """
        Return an iterator over embedding tables, yielding both the table name as well as the embedding
        table itself. The embedding table is in the form of PartiallyMaterializedTensor with a valid
        RocksDB snapshot to support windowed access.
        """
        for emb_module in self._emb_modules:
            if isinstance(emb_module, KeyValueEmbeddingBag) or isinstance(
                emb_module, ZeroCollisionKeyValueEmbeddingBag
            ):
                yield from emb_module.get_named_split_embedding_weights_snapshot()

    def flush(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.flush()

    def create_rocksdb_hard_link_snapshot(self) -> None:
        for emb_module in self._emb_modules:
            if isinstance(emb_module, KeyValueEmbedding):
                emb_module.create_rocksdb_hard_link_snapshot()

    def purge(self) -> None:
        for emb_module in self._emb_modules:
            emb_module.purge()

    def register_optim_state_tracker_fn(
        self,
        record_fn: Callable[
            [KeyedJaggedTensor, torch.Tensor, Optional[nn.Module]], None
        ],
    ) -> None:
        """
        Model tracker function to tracker optimizer state

         Args:
             record_fn (Callable[[KeyedJaggedTensor, torch.Tensor, Optional[nn.Module]], None]): A custom record function to be called after lookup is done.

        """
        self.optim_state_tracker_fn = record_fn


class MetaInferGroupedEmbeddingsLookup(
    BaseEmbeddingLookup[KeyedJaggedTensor, torch.Tensor], TBEToRegisterMixIn
):
    """
    meta embedding lookup module for inference since inference lookup has references
    for multiple TBE ops over all gpu workers.
    inference grouped embedding lookup module contains meta modules allocated over gpu workers.
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        shard_index: Optional[int] = None,
    ) -> None:
        # TODO rename to _create_embedding_kernel
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
            fused_params: Optional[Dict[str, Any]] = None,
            shard_index: Optional[int] = None,
        ) -> BaseBatchedEmbedding[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ]:
            return QuantBatchedEmbedding(
                config=config,
                device=device,
                fused_params=fused_params,
                shard_index=shard_index,
            )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(
                _create_lookup(config, device, fused_params, shard_index)
            )

        self._feature_splits: List[int] = [
            config.num_features() for config in grouped_configs
        ]

        self.grouped_configs = grouped_configs
        self.device: Optional[str] = str(device) if device is not None else None
        self.output_dtype: torch.dtype = (
            fused_params["output_dtype"].as_dtype()
            if fused_params and "output_dtype" in fused_params
            else torch.float16
        )

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return get_tbes_to_register_from_iterable(self._emb_modules)

    def embeddings_cat_empty_rank_handle_inference(
        self,
        embeddings: List[torch.Tensor],
        dim: int = 0,
    ) -> torch.Tensor:
        if len(self.grouped_configs) == 0:
            # return a dummy empty tensor when grouped_configs is empty
            dev: Optional[torch.device] = (
                torch.device(self.device) if self.device is not None else None
            )
            return torch.empty([0], dtype=self.output_dtype, device=dev)
        elif len(self.grouped_configs) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings, dim=dim)

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        features_by_group = (
            [sparse_features]
            if len(self._feature_splits) == 1
            else sparse_features.split(
                self._feature_splits,
            )
        )
        for i in range(len(self._emb_modules)):
            # 2d embedding by nature
            embeddings.append(self._emb_modules[i].forward(features_by_group[i]))

        return self.embeddings_cat_empty_rank_handle_inference(embeddings)

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "MetaInferGroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "MetaInferGroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def flush(self) -> None:
        # not implemented
        pass

    def purge(self) -> None:
        # not implemented
        pass


class MetaInferGroupedPooledEmbeddingsLookup(
    BaseEmbeddingLookup[KeyedJaggedTensor, torch.Tensor], TBEToRegisterMixIn
):
    """
    meta embedding bag lookup module for inference since inference lookup has references
    for multiple TBE ops over all gpu workers.
    inference grouped embedding bag lookup module contains meta modules allocated over gpu workers.
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        shard_index: Optional[int] = None,
    ) -> None:
        # TODO rename to _create_embedding_kernel
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
            fused_params: Optional[Dict[str, Any]] = None,
            shard_index: Optional[int] = None,
        ) -> BaseBatchedEmbeddingBag[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ]:
            return QuantBatchedEmbeddingBag(
                config=config,
                device=device,
                fused_params=fused_params,
                shard_index=shard_index,
            )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(
                _create_lookup(config, device, fused_params, shard_index)
            )

        self._feature_splits: List[int] = [
            config.num_features() for config in grouped_configs
        ]

        self.grouped_configs = grouped_configs
        self._feature_processor = feature_processor
        self.device: Optional[str] = str(device) if device is not None else None
        self.output_dtype: torch.dtype = (
            fused_params["output_dtype"].as_dtype()
            if fused_params and "output_dtype" in fused_params
            else torch.float16
        )

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return get_tbes_to_register_from_iterable(self._emb_modules)

    def embeddings_cat_empty_rank_handle_inference(
        self,
        embeddings: List[torch.Tensor],
        dim: int = 0,
    ) -> torch.Tensor:
        if len(self.grouped_configs) == 0:
            # return a dummy empty tensor when grouped_configs is empty
            dev: Optional[torch.device] = (
                torch.device(self.device) if self.device is not None else None
            )
            return torch.empty([0], dtype=self.output_dtype, device=dev)
        elif len(self.grouped_configs) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings, dim=dim)

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        if len(self.grouped_configs) == 0:
            # return a dummy empty tensor when grouped_configs is empty
            return dummy_tensor(
                sparse_features,
                self.output_dtype,
            )

        embeddings: List[torch.Tensor] = []
        features_by_group = (
            [sparse_features]
            if len(self._feature_splits) == 1
            else sparse_features.split(
                self._feature_splits,
            )
        )
        # syntax for torchscript
        for i, (config, emb_op) in enumerate(
            zip(self.grouped_configs, self._emb_modules)
        ):
            features = features_by_group[i]
            if (
                config.has_feature_processor
                and self._feature_processor is not None
                and isinstance(self._feature_processor, BaseGroupedFeatureProcessor)
            ):
                features = self._feature_processor(features)
            embeddings.append(emb_op.forward(features))

        return self.embeddings_cat_empty_rank_handle_inference(
            embeddings,
            dim=1,
        )

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "MetaInferGroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "MetaInferGroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def flush(self) -> None:
        # not implemented
        pass

    def purge(self) -> None:
        # not implemented
        pass


class InferGroupedLookupMixin(ABC):
    def forward(
        self,
        input_dist_outputs: InputDistOutputs,
    ) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        sparse_features = input_dist_outputs.features
        # syntax for torchscript
        for i, embedding_lookup in enumerate(
            self._embedding_lookups_per_rank,
        ):
            sparse_features_rank = sparse_features[i]
            embeddings.append(embedding_lookup.forward(sparse_features_rank))
        return embeddings

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for rank_modules in self._embedding_lookups_per_rank:
            rank_modules.state_dict(destination, prefix, keep_vars)

        return destination

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        for rank_modules in self._embedding_lookups_per_rank:
            incompatible_keys = rank_modules.load_state_dict(state_dict)
            missing_keys.extend(incompatible_keys.missing_keys)
            unexpected_keys.extend(incompatible_keys.unexpected_keys)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for rank_modules in self._embedding_lookups_per_rank:
            yield from rank_modules.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for rank_modules in self._embedding_lookups_per_rank:
            yield from rank_modules.named_buffers(prefix, recurse)


class InferGroupedPooledEmbeddingsLookup(
    InferGroupedLookupMixin,
    BaseEmbeddingLookup[InputDistOutputs, List[torch.Tensor]],
    TBEToRegisterMixIn,
):
    def __init__(
        self,
        grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]],
        world_size: int,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__()
        self._embedding_lookups_per_rank: List[
            MetaInferGroupedPooledEmbeddingsLookup
        ] = []
        if get_propogate_device():
            device_type: str = (
                "cpu" if device is None else device.type
            )  # TODO: replace hardcoded cpu with DEFAULT_DEVICE_TYPE in torchrec.distributed.types when torch package issue resolved
        else:
            device_type = (
                "meta" if device is not None and device.type == "meta" else "cuda"
            )

        self._is_empty_rank: List[bool] = []
        for rank in range(world_size):
            empty_rank = len(grouped_configs_per_rank[rank]) == 0
            grouped_configs_per_rank_elem = grouped_configs_per_rank[rank]
            contains_virtual_table = any(
                config.is_using_virtual_table()
                for config in grouped_configs_per_rank_elem
            )
            # In case of heterogenous sharding of a single table acorss
            # different device types i.e. when device_type_from_sharding_infos
            # is a tuple OR if any of the table is virtual table, we can for
            # now assume that the table is row_wise sharded and the shard_index
            # can be set to the rank. shard_index is used downstream to get
            # runtime_device (or row alignment) as well as to get the shard
            # offsets for virtual table
            shard_index = (
                rank
                if isinstance(device_type_from_sharding_infos, tuple)
                or contains_virtual_table
                else None
            )
            self._is_empty_rank.append(empty_rank)
            if not empty_rank:
                self._embedding_lookups_per_rank.append(
                    # TODO add position weighted module support
                    MetaInferGroupedPooledEmbeddingsLookup(
                        grouped_configs=grouped_configs_per_rank[rank],
                        device=rank_device(device_type, rank),
                        fused_params=fused_params,
                        shard_index=shard_index,
                    )
                )

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return get_tbes_to_register_from_iterable(self._embedding_lookups_per_rank)

    def forward(
        self,
        input_dist_outputs: InputDistOutputs,
    ) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        sparse_features = [
            input_dist_outputs.features[i]
            for i, is_empty in enumerate(self._is_empty_rank)
            if not is_empty
        ]
        # syntax for torchscript
        for i, embedding_lookup in enumerate(
            self._embedding_lookups_per_rank,
        ):
            sparse_features_rank = sparse_features[i]
            embeddings.append(embedding_lookup.forward(sparse_features_rank))
        return embeddings


class InferGroupedEmbeddingsLookup(
    InferGroupedLookupMixin,
    BaseEmbeddingLookup[InputDistOutputs, List[torch.Tensor]],
    TBEToRegisterMixIn,
):
    def __init__(
        self,
        grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]],
        world_size: int,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__()
        self._embedding_lookups_per_rank: List[MetaInferGroupedEmbeddingsLookup] = []

        if get_propogate_device():
            device_type: str = (
                "cpu" if device is None else device.type
            )  # TODO: replace hardcoded cpu with DEFAULT_DEVICE_TYPE in torchrec.distributed.types when torch package issue resolved
        else:
            device_type = (
                "meta" if device is not None and device.type == "meta" else "cuda"
            )
        for rank in range(world_size):
            grouped_configs_per_rank_elem = grouped_configs_per_rank[rank]
            contains_virtual_table = any(
                config.is_using_virtual_table()
                for config in grouped_configs_per_rank_elem
            )
            # In case of heterogenous sharding of a single table acorss
            # different device types i.e. when device_type_from_sharding_infos
            # is a tuple OR if any of the table is virtual table, we can for
            # now assume that the table is row_wise sharded and the shard_index
            # can be set to the rank. shard_index is used downstream to get
            # runtime_device (or row alignment) as well as to get the shard
            # offsets for virtual table
            shard_index = (
                rank
                if isinstance(device_type_from_sharding_infos, tuple)
                or contains_virtual_table
                else None
            )
            device = rank_device(device_type, rank)
            self._embedding_lookups_per_rank.append(
                MetaInferGroupedEmbeddingsLookup(
                    grouped_configs=grouped_configs_per_rank[rank],
                    device=rank_device(device_type, rank),
                    fused_params=fused_params,
                    shard_index=shard_index,
                )
            )

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return get_tbes_to_register_from_iterable(self._embedding_lookups_per_rank)
