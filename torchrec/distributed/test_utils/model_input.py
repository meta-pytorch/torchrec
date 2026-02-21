#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from dataclasses import dataclass, field
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
    dummy: List[torch.Tensor] = field(default_factory=list)

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        data_copy_stream: Optional[torch.cuda.streams.Stream] = None,
    ) -> "ModelInput":
        """
        Move ModelInput to the specified device.

        Args:
            device: Target device to move tensors to.
            non_blocking: Whether to perform asynchronous copies.
            data_copy_stream: Optional CUDA stream for async data copies. When provided,
                tensors are pre-allocated on the target device and copied within this stream.
                This enables pipelined data transfers with computation on other streams.

        Returns:
            ModelInput on the target device.

        Example:
            # Standard synchronous transfer
            batch_gpu = batch_cpu.to(device="cuda")

            # Async transfer with dedicated stream
            copy_stream = torch.cuda.Stream()
            batch_gpu = batch_cpu.to(device="cuda", non_blocking=True, data_copy_stream=copy_stream)
        """
        if data_copy_stream is None:
            # Standard .to() method
            float_features = self.float_features.to(
                device=device,
                non_blocking=non_blocking,
            )
            idlist_features = (
                self.idlist_features.to(
                    device=device,
                    non_blocking=non_blocking,
                )
                if self.idlist_features is not None
                else None
            )
            idscore_features = (
                self.idscore_features.to(
                    device=device,
                    non_blocking=non_blocking,
                )
                if self.idscore_features is not None
                else None
            )
            label = self.label.to(
                device=device,
                non_blocking=non_blocking,
            )
            dummy = [d.to(device=device, non_blocking=non_blocking) for d in self.dummy]
        else:
            # Async copy using dedicated stream
            current_stream = torch.cuda.current_stream(device)

            # Pre-allocate tensors on target device
            float_features = torch.empty_like(self.float_features, device=device)
            label = torch.empty_like(self.label, device=device)
            idlist_features = (
                None
                if self.idlist_features is None
                else KeyedJaggedTensor.empty_like(self.idlist_features, device=device)
            )
            idscore_features = (
                None
                if self.idscore_features is None
                else KeyedJaggedTensor.empty_like(self.idscore_features, device=device)
            )
            dummy = [torch.empty_like(d, device=device) for d in self.dummy]

            # Perform async copy in dedicated stream
            with data_copy_stream:
                # Wait for current stream to finish memory allocation
                data_copy_stream.wait_stream(current_stream)

                float_features.copy_(self.float_features, non_blocking=non_blocking)
                label.copy_(self.label, non_blocking=non_blocking)
                if idlist_features is not None:
                    idlist_features.copy_(
                        # pyrefly: ignore[bad-argument-type]
                        self.idlist_features,
                        non_blocking=non_blocking,
                    )
                if idscore_features is not None:
                    idscore_features.copy_(
                        # pyrefly: ignore[bad-argument-type]
                        self.idscore_features,
                        non_blocking=non_blocking,
                    )
                dummy = [
                    d.copy_(self.dummy[i], non_blocking=non_blocking)
                    for (i, d) in enumerate(dummy)
                ]

        return ModelInput(
            float_features=float_features,
            idlist_features=idlist_features,
            idscore_features=idscore_features,
            label=label,
            dummy=dummy,
        )

    def record_stream(self, stream: torch.Stream) -> None:
        """
        need to explicitly call `record_stream` for non-pytorch native object (KJT)
        """
        self.float_features.record_stream(stream)
        if isinstance(self.idlist_features, KeyedJaggedTensor):
            # pyrefly: ignore[bad-argument-type]
            self.idlist_features.record_stream(stream)
        if isinstance(self.idscore_features, KeyedJaggedTensor):
            # pyrefly: ignore[bad-argument-type]
            self.idscore_features.record_stream(stream)
        self.label.record_stream(stream)
        for d in self.dummy:
            d.record_stream(stream)

    def size_in_bytes(self) -> int:
        """
        Returns the size of the ModelInput in bytes.
        Recursively computes size for all contained tensors and sparse data structures.
        """
        size = self.float_features.element_size() * self.float_features.numel()
        size += self.label.element_size() * self.label.numel()
        if self.idlist_features is not None:
            size += self.idlist_features.size_in_bytes()
        if self.idscore_features is not None:
            size += self.idscore_features.size_in_bytes()
        for d in self.dummy:
            size += d.element_size() * d.numel()
        return size

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
        num_dummy_tensor: int = 0,
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
        dummy_list = [
            [
                torch.rand((batch_size, num_float_features), device=device)
                for _ in range(num_dummy_tensor)
            ]
            for _ in range(world_size)
        ]
        global_input = ModelInput(
            float_features=torch.cat(float_features_list),
            idlist_features=global_idlist_features,
            idscore_features=global_idscore_features,
            label=torch.cat(label_list),
            dummy=[
                torch.cat([dummy_list[r][i] for r in range(world_size)])
                for i in range(num_dummy_tensor)
            ],
        )
        local_inputs = [
            ModelInput(
                float_features=float_features,
                idlist_features=idlist_features,
                idscore_features=idscore_features,
                label=label,
                dummy=dummy,
            )
            for float_features, idlist_features, idscore_features, label, dummy in zip(
                float_features_list,
                idlist_features_list,
                idscore_features_list,
                label_list,
                dummy_list,
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
        num_dummy_tensor: int = 0,
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
                num_dummy_tensor=num_dummy_tensor,
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
        power_law_alpha: Optional[
            float
        ] = None,  # If set, use power-law distribution for indices
        num_dummy_tensor: int = 0,
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
                power_law_alpha=power_law_alpha,
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
                weighted=True,  # weighted
                max_feature_lengths=max_feature_lengths,
                use_offsets=use_offsets,
                device=device,
                indices_dtype=indices_dtype,
                offsets_dtype=offsets_dtype,
                lengths_dtype=lengths_dtype,
                all_zeros=all_zeros,
                power_law_alpha=power_law_alpha,
            )
            if weighted_tables is not None and len(weighted_tables) > 0
            else None
        )
        label = (
            torch.zeros((batch_size,), device=device)
            if all_zeros
            else torch.rand((batch_size,), device=device)
        )
        dummy = [
            torch.rand((batch_size, num_float_features), device=device)
            for _ in range(num_dummy_tensor)
        ]
        if pin_memory:
            float_features, idlist_features, idscore_features, label, dummy = (
                ModelInput._pin_memory(
                    float_features, idlist_features, idscore_features, label, dummy
                )
            )

        return ModelInput(
            float_features=float_features,
            idlist_features=idlist_features,
            idscore_features=idscore_features,
            label=label,
            dummy=dummy,
        )

    @staticmethod
    def _generate_power_law_indices(
        alpha: float,
        num_indices: int,
        num_embeddings: int,
        dtype: torch.dtype,
        device: Optional[torch.device],
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate indices following a power-law distribution.

        For a continuous power-law distribution f(x) ∝ 1/x^alpha on [1, n],
        this uses inverse CDF sampling and shifts results to produce 0-indexed
        outputs in [0, n-1].

        Args:
            alpha: The power-law exponent (must be >= 0). Higher values produce more
                skewed distributions with more samples at low indices.
                - alpha=0: uniform distribution
                - 0<alpha<1: truncated power-law via inverse CDF
                - alpha≈1: log-uniform distribution (special case, uses tolerance)
                - alpha>1: Pareto distribution via inverse CDF with rejection for truncation
            num_indices: Number of indices to generate.
            num_embeddings: Maximum index value (exclusive), i.e., indices in [0, num_embeddings).
                Must be >= 1.
            dtype: Data type of the output tensor.
            device: Device to generate tensor on.
            seed: Optional random seed (unused, for API compatibility).

        Returns:
            Tensor of indices following the power-law distribution, in range [0, num_embeddings).

        Raises:
            ValueError: If alpha < 0 or num_embeddings < 1.
        """
        # Validate inputs
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if num_embeddings < 1:
            raise ValueError(f"num_embeddings must be >= 1, got {num_embeddings}")

        # Handle trivial case: only one possible index
        if num_embeddings == 1:
            return torch.zeros(num_indices, dtype=dtype, device=device)

        if alpha == 0.0:
            return torch.randint(
                0, num_embeddings, (num_indices,), dtype=dtype, device=device
            )

        u = torch.rand(num_indices, device=device)
        # Avoid u=0 or u=1 which can cause inf
        u = u.clamp(1e-10, 1 - 1e-10)

        # Use tolerance for alpha ≈ 1 to avoid numerical instability
        # When |alpha - 1| < tolerance, exponents become very large (>500)
        # Using 2e-3 to account for floating-point representation issues
        # (e.g., abs(0.999 - 1.0) may be slightly > 1e-3 due to float precision)
        alpha_tolerance = 2e-3
        if abs(alpha - 1.0) < alpha_tolerance:
            # Log-uniform distribution (f(x) ∝ 1/x)
            # CDF: F(x) = ln(x) / ln(n)
            # Inverse CDF: x = n^u, produces values in [1, n]
            # Subtract 1 to convert from 1-indexed to 0-indexed
            indices = (num_embeddings**u - 1).long()
        elif alpha < 1.0:
            # Truncated power-law on [1, n] with f(x) ∝ 1/x^alpha
            # CDF: F(x) = (x^(1-alpha) - 1) / (n^(1-alpha) - 1)
            # Inverse CDF: x = (u * (n^(1-alpha) - 1) + 1)^(1/(1-alpha))
            # Subtract 1 to convert from 1-indexed [1,n] to 0-indexed [0,n-1]
            n_term = num_embeddings ** (1 - alpha) - 1
            indices = ((u * n_term + 1) ** (1 / (1 - alpha)) - 1).long()
        else:
            # Pareto/power-law on [1, inf) with f(x) ∝ 1/x^alpha, alpha > 1
            # CDF: F(x) = 1 - x^(1-alpha)
            # Inverse CDF: x = (1-u)^(1/(1-alpha)) = (1-u)^(-1/(alpha-1))
            # Subtract 1 to convert from 1-indexed to 0-indexed
            exponent = -1 / (alpha - 1)
            indices = ((1 - u) ** exponent - 1).long()
            # Resample any out-of-bounds values
            mask = indices >= num_embeddings
            while mask.any():
                u_new = torch.rand(mask.sum(), device=device).clamp(1e-10, 1 - 1e-10)
                indices[mask] = ((1 - u_new) ** exponent - 1).long()
                mask = indices >= num_embeddings

        return indices.clamp(0, num_embeddings - 1).to(
            dtype
        )  # safety clamp, shouldn't trigger

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
        power_law_alpha: Optional[float] = None,
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
        # pyrefly: ignore[bad-argument-type]
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
            if all_zeros:
                _indices = torch.zeros(
                    (num_indices,),
                    dtype=indices_dtype,
                    device=device,
                )
            elif power_law_alpha is not None:
                _indices = ModelInput._generate_power_law_indices(
                    alpha=power_law_alpha,
                    num_indices=num_indices,
                    num_embeddings=num_embeddings,
                    dtype=indices_dtype,
                    device=device,
                )
            else:
                _indices = torch.randint(
                    0,
                    num_embeddings,
                    (num_indices,),
                    dtype=indices_dtype,
                    device=device,
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
    def _pin_memory(
        float_features: torch.Tensor,
        idlist_features: Optional[KeyedJaggedTensor],
        idscore_features: Optional[KeyedJaggedTensor],
        label: torch.Tensor,
        dummy: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[KeyedJaggedTensor],
        Optional[KeyedJaggedTensor],
        torch.Tensor,
        List[torch.Tensor],
    ]:
        """
        Pin memory for all tensors in `ModelInput`

        All tensors in `ModelInput` should be on pinned memory otherwise
        the `_to_copy` (host-to-device) data transfer still blocks cpu execution
        """
        return (
            float_features.pin_memory(),
            idlist_features.pin_memory() if idlist_features is not None else None,
            idscore_features.pin_memory() if idscore_features is not None else None,
            label.pin_memory(),
            [d.pin_memory() for d in dummy] if dummy else [],
        )

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
        power_law_alpha: Optional[float] = None,
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
                power_law_alpha=power_law_alpha,
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


@dataclass
class VariableBatchModelInput(ModelInput):

    float_features: torch.Tensor
    idlist_features: Optional[KeyedJaggedTensor]
    idscore_features: Optional[KeyedJaggedTensor]
    label: torch.Tensor

    @classmethod
    # pyrefly: ignore[bad-param-name-override]
    def generate(
        cls,
        batch_size: int = 1,
        num_float_features: int = 16,
        dedup_factor: int = 2,
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
        pooling_avg: int = 10,
        tables_pooling: Optional[List[int]] = None,
        max_feature_lengths: Optional[List[int]] = None,
        use_offsets: bool = False,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        all_zeros: bool = False,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,  # pin_memory is needed for training job qps benchmark
        num_dummy_tensor: int = 0,
    ) -> "VariableBatchModelInput":
        """
        Returns a single batch of `VariableBatchModelInput`

        Different from `ModelInput`, `batch_size` is the average batch size which
        is used together with the `dedup_factor` to get the actual batch size.
        """

        float_features = torch.rand(
            (dedup_factor * batch_size, num_float_features), device=device
        )

        idlist_features = (
            VariableBatchModelInput._create_variable_batch_kjt(
                tables=tables,
                average_batch_size=batch_size,
                dedup_factor=dedup_factor,
                use_offsets=use_offsets,
                indices_dtype=indices_dtype,
                offsets_dtype=offsets_dtype,
                lengths_dtype=lengths_dtype,
                device=device,
            )
            if tables is not None and len(tables) > 0
            else None
        )

        idscore_features = (
            VariableBatchModelInput._create_variable_batch_kjt(
                tables=weighted_tables,
                average_batch_size=batch_size,
                dedup_factor=dedup_factor,
                use_offsets=use_offsets,
                indices_dtype=indices_dtype,
                offsets_dtype=offsets_dtype,
                lengths_dtype=lengths_dtype,
                device=device,
            )
            if weighted_tables is not None and len(weighted_tables) > 0
            else None
        )

        label = torch.rand((dedup_factor * batch_size), device=device)

        dummy = [
            torch.rand((dedup_factor * batch_size, num_float_features), device=device)
            for _ in range(num_dummy_tensor)
        ]

        if pin_memory:
            float_features, idlist_features, idscore_features, label, dummy = (
                ModelInput._pin_memory(
                    float_features, idlist_features, idscore_features, label, dummy
                )
            )

        return VariableBatchModelInput(
            float_features=float_features,
            idlist_features=idlist_features,
            idscore_features=idscore_features,
            label=label,
            dummy=dummy,
        )

    @staticmethod
    def _create_variable_batch_kjt(
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        average_batch_size: int,
        dedup_factor: int,
        use_offsets: bool = False,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        device: Optional[torch.device] = None,
    ) -> KeyedJaggedTensor:

        is_weighted = (
            True if tables and getattr(tables[0], "is_weighted", False) else False
        )

        feature_num_embeddings = {}
        for table in tables:
            for feature_name in table.feature_names:
                feature_num_embeddings[feature_name] = (
                    table.num_embeddings_post_pruning
                    if table.num_embeddings_post_pruning
                    else table.num_embeddings
                )

        keys = list(feature_num_embeddings.keys())
        lengths_per_feature = {}
        values_per_feature = {}
        strides_per_feature = {}
        inverse_indices_per_feature = {}
        weights_per_feature = {} if is_weighted else None

        for key, num_embeddings in feature_num_embeddings.items():
            batch_size = random.randint(1, average_batch_size * dedup_factor - 1)
            lengths = torch.randint(
                low=0,
                high=5,
                size=(batch_size,),
                dtype=lengths_dtype,
                device=device,
            )
            lengths_per_feature[key] = lengths
            lengths_sum = sum(lengths.tolist())
            values = torch.randint(
                0,
                num_embeddings,
                (lengths_sum,),
                dtype=indices_dtype,
                device=device,
            )
            values_per_feature[key] = values
            if weights_per_feature is not None:
                weights_per_feature[key] = torch.rand(
                    lengths_sum,
                    device=device,
                )
            strides_per_feature[key] = batch_size
            inverse_indices_per_feature[key] = torch.randint(
                0,
                batch_size,
                (dedup_factor * average_batch_size,),
                dtype=indices_dtype,
                device=device,
            )

        values = torch.cat(list(values_per_feature.values()))
        lengths = torch.cat(list(lengths_per_feature.values()))
        weights = (
            torch.cat(list(weights_per_feature.values()))
            if weights_per_feature is not None
            else None
        )
        inverse_indices = (
            keys,
            torch.stack(list(inverse_indices_per_feature.values())),
        )
        strides = [[stride] for stride in strides_per_feature.values()]

        if use_offsets:
            offsets = torch.cat(
                [
                    torch.tensor(
                        [0],
                        dtype=offsets_dtype,
                        device=device,
                    ),
                    lengths.cumsum(0),
                ]
            )
            return KeyedJaggedTensor(
                keys=keys,
                values=values,
                offsets=offsets,
                weights=weights,
                stride_per_key_per_rank=strides,
                inverse_indices=inverse_indices,
            )

        return KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            weights=weights,
            stride_per_key_per_rank=strides,
            inverse_indices=inverse_indices,
        )


@dataclass
class TdModelInput(ModelInput):
    # pyrefly: ignore[bad-override]
    idlist_features: TensorDict
