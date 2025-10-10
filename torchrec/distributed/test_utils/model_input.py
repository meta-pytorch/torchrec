#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import cast, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class ModelInput(Pipelineable):
    """
    basic model input for a simple standard RecSys model
    the input is a training data batch that contains:
    1. a tensor for dense features
    2. a KJT for unweighted sparse features
    3. a KJT for weighted sparse features
    4. a tensor for the label
    """

    float_features: torch.Tensor
    idlist_features: Optional[KeyedJaggedTensor]
    idscore_features: Optional[KeyedJaggedTensor]
    label: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "ModelInput":
        return ModelInput(
            float_features=self.float_features.to(
                device=device, non_blocking=non_blocking
            ),
            idlist_features=(
                self.idlist_features.to(device=device, non_blocking=non_blocking)
                if self.idlist_features is not None
                else None
            ),
            idscore_features=(
                self.idscore_features.to(device=device, non_blocking=non_blocking)
                if self.idscore_features is not None
                else None
            ),
            label=self.label.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.Stream) -> None:
        """
        need to explicitly call `record_stream` for non-pytorch native object (KJT)
        """
        self.float_features.record_stream(stream)
        if isinstance(self.idlist_features, KeyedJaggedTensor):
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            self.idlist_features.record_stream(stream)
        if isinstance(self.idscore_features, KeyedJaggedTensor):
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            self.idscore_features.record_stream(stream)
        self.label.record_stream(stream)

    @classmethod
    def generate_global_and_local_batches(
        cls,
        world_size: int,
        batch_size: int = 1,
        tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        weighted_tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        num_float_features: int = 16,
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        max_feature_lengths: Optional[List[int]] = None,
        use_offsets: bool = False,
        device: Optional[torch.device] = None,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
    ) -> Tuple["ModelInput", List["ModelInput"]]:
        """
        Returns a global (single-rank training) batch, and a list of local
        (multi-rank training) batches of world_size. The data should be
        consistent between the local batches and the global batch so that
        they can be used for comparison and validation.
        """

        float_features_list = [
            (
                torch.zeros((batch_size, num_float_features), device=device)
                if all_zeros
                else torch.rand((batch_size, num_float_features), device=device)
            )
            for _ in range(world_size)
        ]
        global_idlist_features, idlist_features_list = (
            ModelInput._create_batched_standard_kjts(
                batch_size,
                world_size,
                tables,
                pooling_avg,
                tables_pooling,
                False,  # unweighted
                max_feature_lengths,
                use_offsets,
                device,
                indices_dtype,
                offsets_dtype,
                lengths_dtype,
                all_zeros,
            )
            if tables is not None and len(tables) > 0
            else (None, [None for _ in range(world_size)])
        )
        global_idscore_features, idscore_features_list = (
            ModelInput._create_batched_standard_kjts(
                batch_size,
                world_size,
                weighted_tables,
                pooling_avg,
                tables_pooling,
                True,  # weighted
                max_feature_lengths,
                use_offsets,
                device,
                indices_dtype,
                offsets_dtype,
                lengths_dtype,
                all_zeros,
            )
            if weighted_tables is not None and len(weighted_tables) > 0
            else (None, [None for _ in range(world_size)])
        )
        label_list = [
            (
                torch.zeros((batch_size,), device=device)
                if all_zeros
                else torch.rand((batch_size,), device=device)
            )
            for _ in range(world_size)
        ]
        global_input = ModelInput(
            float_features=torch.cat(float_features_list),
            idlist_features=global_idlist_features,
            idscore_features=global_idscore_features,
            label=torch.cat(label_list),
        )
        local_inputs = [
            ModelInput(
                float_features=float_features,
                idlist_features=idlist_features,
                idscore_features=idscore_features,
                label=label,
            )
            for float_features, idlist_features, idscore_features, label in zip(
                float_features_list,
                idlist_features_list,
                idscore_features_list,
                label_list,
            )
        ]
        return global_input, local_inputs

    @classmethod
    def generate_local_batches(
        cls,
        world_size: int,
        batch_size: int = 1,
        tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        weighted_tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        num_float_features: int = 16,
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        max_feature_lengths: Optional[List[int]] = None,
        use_offsets: bool = False,
        device: Optional[torch.device] = None,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
        pin_memory: bool = False,  # pin_memory is needed for training job qps benchmark
    ) -> List["ModelInput"]:
        """
        Returns multi-rank batches (ModelInput) of world_size
        """
        return [
            cls.generate(
                batch_size=batch_size,
                tables=tables,
                weighted_tables=weighted_tables,
                num_float_features=num_float_features,
                pooling_avg=pooling_avg,
                tables_pooling=tables_pooling,
                max_feature_lengths=max_feature_lengths,
                use_offsets=use_offsets,
                device=device,
                indices_dtype=indices_dtype,
                offsets_dtype=offsets_dtype,
                lengths_dtype=lengths_dtype,
                all_zeros=all_zeros,
                pin_memory=pin_memory,
            )
            for _ in range(world_size)
        ]

    @classmethod
    def generate(
        cls,
        batch_size: int = 1,
        tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        weighted_tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        num_float_features: int = 16,
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        max_feature_lengths: Optional[List[int]] = None,
        use_offsets: bool = False,
        device: Optional[torch.device] = None,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
        pin_memory: bool = False,  # pin_memory is needed for training job qps benchmark
    ) -> "ModelInput":
        """
        Returns a single batch of `ModelInput`

        The `pin_memory()` call for all KJT tensors are important for training benchmark, and
        also valid argument for the prod training scenario: TrainModelInput should be created
        on pinned memory for a fast transfer to gpu. For more on pin_memory:
        https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html#pin-memory
        """
        float_features = (
            torch.zeros((batch_size, num_float_features), device=device)
            if all_zeros
            else torch.rand((batch_size, num_float_features), device=device)
        )
        idlist_features = (
            ModelInput.create_standard_kjt(
                batch_size=batch_size,
                tables=tables,
                pooling_avg=pooling_avg,
                tables_pooling=tables_pooling,
                weighted=False,  # unweighted
                max_feature_lengths=max_feature_lengths,
                use_offsets=use_offsets,
                device=device,
                indices_dtype=indices_dtype,
                offsets_dtype=offsets_dtype,
                lengths_dtype=lengths_dtype,
                all_zeros=all_zeros,
            )
            if tables is not None and len(tables) > 0
            else None
        )
        idscore_features = (
            ModelInput.create_standard_kjt(
                batch_size=batch_size,
                tables=weighted_tables,
                pooling_avg=pooling_avg,
                tables_pooling=tables_pooling,
                weighted=False,  # weighted
                max_feature_lengths=max_feature_lengths,
                use_offsets=use_offsets,
                device=device,
                indices_dtype=indices_dtype,
                offsets_dtype=offsets_dtype,
                lengths_dtype=lengths_dtype,
                all_zeros=all_zeros,
            )
            if weighted_tables is not None and len(weighted_tables) > 0
            else None
        )
        label = (
            torch.zeros((batch_size,), device=device)
            if all_zeros
            else torch.rand((batch_size,), device=device)
        )
        if pin_memory:
            # all tensors in `ModelInput` should be on pinned memory otherwise
            # the `_to_copy` (host-to-device) data transfer still blocks cpu execution
            float_features = float_features.pin_memory()
            label = label.pin_memory()
            idlist_features = (
                None if idlist_features is None else idlist_features.pin_memory()
            )
            idscore_features = (
                None if idscore_features is None else idscore_features.pin_memory()
            )
        return ModelInput(
            float_features=float_features,
            idlist_features=idlist_features,
            idscore_features=idscore_features,
            label=label,
        )

    @staticmethod
    def _create_features_lengths_indices(
        batch_size: int,
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        max_feature_lengths: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        indices_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Create keys, lengths, and indices for a KeyedJaggedTensor from embedding table configs.

        Returns:
            Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
                Feature names, per-feature lengths, and per-feature indices.
        """
        pooling_factor_per_feature: List[int] = []
        num_embeddings_per_feature: List[int] = []
        max_length_per_feature: List[Optional[int]] = []
        features: List[str] = []
        for tid, table in enumerate(tables):
            pooling_factor = (
                tables_pooling[tid] if tables_pooling is not None else pooling_avg
            )
            max_feature_length = (
                max_feature_lengths[tid] if max_feature_lengths is not None else None
            )
            features.extend(table.feature_names)
            for _ in table.feature_names:
                pooling_factor_per_feature.append(pooling_factor)
                num_embeddings_per_feature.append(
                    table.num_embeddings_post_pruning or table.num_embeddings
                )
                max_length_per_feature.append(max_feature_length)

        lengths_per_feature: List[torch.Tensor] = []
        indices_per_feature: List[torch.Tensor] = []

        for pooling_factor, num_embeddings, max_length in zip(
            pooling_factor_per_feature,
            num_embeddings_per_feature,
            max_length_per_feature,
        ):
            # lengths
            _lengths = torch.max(
                torch.normal(
                    pooling_factor,
                    pooling_factor / 10,  # std
                    [batch_size],
                    device=device,
                ),
                torch.tensor(1.0, device=device),
            ).to(lengths_dtype)
            if max_length:
                _lengths = torch.clamp(_lengths, max=max_length)
            lengths_per_feature.append(_lengths)

            # indices
            num_indices = cast(int, torch.sum(_lengths).item())
            _indices = (
                torch.zeros(
                    (num_indices,),
                    dtype=indices_dtype,
                    device=device,
                )
                if all_zeros
                else torch.randint(
                    0,
                    num_embeddings,
                    (num_indices,),
                    dtype=indices_dtype,
                    device=device,
                )
            )
            indices_per_feature.append(_indices)
        return features, lengths_per_feature, indices_per_feature

    @staticmethod
    def _assemble_kjt(
        features: List[str],
        lengths_per_feature: List[torch.Tensor],
        indices_per_feature: List[torch.Tensor],
        weighted: bool = False,
        device: Optional[torch.device] = None,
        use_offsets: bool = False,
        offsets_dtype: torch.dtype = torch.int64,
    ) -> KeyedJaggedTensor:
        """
        Assembles a KeyedJaggedTensor (KJT) from the provided per-feature lengths and indices.

        This method is used to generate corresponding local_batches and global_batch KJTs.
        It concatenates the lengths and indices for each feature to form a complete KJT.
        """

        lengths = torch.cat(lengths_per_feature)
        indices = torch.cat(indices_per_feature)
        offsets = None
        weights = torch.rand((indices.numel(),), device=device) if weighted else None
        if use_offsets:
            offsets = torch.cat(
                [torch.tensor([0], device=device), lengths.cumsum(0)]
            ).to(offsets_dtype)
            lengths = None
        return KeyedJaggedTensor(features, indices, weights, lengths, offsets)

    @staticmethod
    def create_standard_kjt(
        batch_size: int,
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        weighted: bool = False,
        max_feature_lengths: Optional[List[int]] = None,
        use_offsets: bool = False,
        device: Optional[torch.device] = None,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
    ) -> KeyedJaggedTensor:
        features, lengths_per_feature, indices_per_feature = (
            ModelInput._create_features_lengths_indices(
                batch_size=batch_size,
                tables=tables,
                pooling_avg=pooling_avg,
                tables_pooling=tables_pooling,
                max_feature_lengths=max_feature_lengths,
                device=device,
                indices_dtype=indices_dtype,
                lengths_dtype=lengths_dtype,
                all_zeros=all_zeros,
            )
        )
        return ModelInput._assemble_kjt(
            features=features,
            lengths_per_feature=lengths_per_feature,
            indices_per_feature=indices_per_feature,
            weighted=weighted,
            device=device,
            use_offsets=use_offsets,
            offsets_dtype=offsets_dtype,
        )

    @staticmethod
    def _create_batched_standard_kjts(
        batch_size: int,
        world_size: int,
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        weighted: bool = False,
        max_feature_lengths: Optional[List[int]] = None,
        use_offsets: bool = False,
        device: Optional[torch.device] = None,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
    ) -> Tuple[KeyedJaggedTensor, List[KeyedJaggedTensor]]:
        """
        generate a global KJT and corresponding per-rank KJTs, the data are the same
        so that they can be used for result comparison.
        """
        data_per_rank = [
            ModelInput._create_features_lengths_indices(
                batch_size,
                tables,
                pooling_avg,
                tables_pooling,
                max_feature_lengths,
                device,
                indices_dtype,
                lengths_dtype,
                all_zeros,
            )
            for _ in range(world_size)
        ]
        features = data_per_rank[0][0]
        local_kjts = [
            ModelInput._assemble_kjt(
                features,
                lengths_per_feature,
                indices_per_feature,
                weighted,
                device,
                use_offsets,
                offsets_dtype,
            )
            for _, lengths_per_feature, indices_per_feature in data_per_rank
        ]
        global_lengths = [
            data_per_rank[r][1][f]
            for f in range(len(features))
            for r in range(world_size)
        ]
        global_indices = [
            data_per_rank[r][2][f]
            for f in range(len(features))
            for r in range(world_size)
        ]
        global_kjt = ModelInput._assemble_kjt(
            features,
            global_lengths,
            global_indices,
            weighted,
            device,
            use_offsets,
            offsets_dtype,
        )
        return global_kjt, local_kjts


# @dataclass
# class VbModelInput(ModelInput):
#     pass

#     @staticmethod
#     def _create_variable_batch_kjt() -> KeyedJaggedTensor:
#         pass

#     @staticmethod
#     def _merge_variable_batch_kjts(kjts: List[KeyedJaggedTensor]) -> KeyedJaggedTensor:
#         pass


@dataclass
class TdModelInput(ModelInput):
    idlist_features: TensorDict  # pyre-ignore
