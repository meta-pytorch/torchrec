#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from dataclasses import dataclass, field
from typing import cast, Dict, List, Optional, Set, Tuple, Union

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
        data_copy_stream: Optional[torch.Stream] = None,
        dense_only: bool = False,
    ) -> "ModelInput":
        """
        Move ModelInput to the specified device.

        Args:
            device: Target device to move tensors to.
            non_blocking: Whether to perform asynchronous copies.
            data_copy_stream: Optional device stream for async data copies. When provided,
                tensors are pre-allocated on the target device and copied within this stream.
                This enables pipelined data transfers with computation on other streams.
            dense_only: Whether to only move dense features (float_features, label, dummy)
                to the target device, keeping sparse features (idlist_features, idscore_features)
                on their current device.

        Returns:
            ModelInput on the target device.

        Example:
            # Standard synchronous transfer
            batch_gpu = batch_cpu.to(device="cuda")

            # Async transfer with dedicated stream
            copy_stream = torch.Stream(device_type="cuda")
            batch_gpu = batch_cpu.to(device="cuda", non_blocking=True, data_copy_stream=copy_stream)

            # Move only dense features to GPU, keep sparse on CPU
            batch_mixed = batch_cpu.to(device="cuda", dense_only=True)
        """
        if data_copy_stream is None:
            # Standard .to() method
            float_features = self.float_features.to(
                device=device,
                non_blocking=non_blocking,
            )
            label = self.label.to(
                device=device,
                non_blocking=non_blocking,
            )
            dummy = [d.to(device=device, non_blocking=non_blocking) for d in self.dummy]

            if dense_only:
                # Keep sparse features on their current device
                idlist_features = self.idlist_features
                idscore_features = self.idscore_features
            else:
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
        else:
            # Async copy using dedicated stream
            device_module = torch.get_device_module(device)
            current_stream = device_module.current_stream(device)

            # Pre-allocate dense tensors on target device
            float_features = torch.empty_like(self.float_features, device=device)
            label = torch.empty_like(self.label, device=device)
            dummy = [torch.empty_like(d, device=device) for d in self.dummy]

            if dense_only:
                # Keep sparse features on their current device
                idlist_features = self.idlist_features
                idscore_features = self.idscore_features
            else:
                # Pre-allocate sparse tensors on target device
                idlist_features = (
                    None
                    if self.idlist_features is None
                    else KeyedJaggedTensor.empty_like(
                        self.idlist_features, device=device
                    )
                )
                idscore_features = (
                    None
                    if self.idscore_features is None
                    else KeyedJaggedTensor.empty_like(
                        self.idscore_features, device=device
                    )
                )

            # Perform async copy in dedicated stream
            with device_module.stream(data_copy_stream):
                # Wait for current stream to finish memory allocation
                data_copy_stream.wait_stream(current_stream)

                # Copy dense features
                float_features.copy_(self.float_features, non_blocking=non_blocking)
                label.copy_(self.label, non_blocking=non_blocking)
                dummy = [
                    d.copy_(self.dummy[i], non_blocking=non_blocking)
                    for (i, d) in enumerate(dummy)
                ]

                # Copy sparse features only if not dense_only
                if not dense_only:
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
                # pyrefly: ignore [no-matching-overload]
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


def _generate_overlap_indices(
    total_indices: int,
    index_range: int,
    overlap_ratio: float,
    prev_unique_indices: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates indices in [0, index_range) with controlled overlap vs the prev batch.

    Overlap ratio is defined as |S_t intersect S_{t+1}| / |S_t|. Fresh indices are
    drawn exclusively from the complement of prev_unique_indices (via a boolean
    mask) to avoid accidental overlap.

    Args:
        total_indices: number of indices to generate.
        index_range: exclusive upper bound of the index range.
        overlap_ratio: target fraction of prev unique indices to reuse.
        prev_unique_indices: unique indices from the previous batch, or None.
        device: device for the output tensors.
        dtype: dtype for the output tensors.

    Returns:
        (generated indices, unique indices in the new batch).
    """
    if total_indices == 0:
        empty = torch.empty(0, dtype=dtype, device=device)
        return empty, empty

    if prev_unique_indices is None or prev_unique_indices.numel() == 0:
        indices = torch.randint(
            0, index_range, (total_indices,), dtype=dtype, device=device
        )
        return indices, indices.unique()

    num_prev = len(prev_unique_indices)
    num_reuse = min(round(overlap_ratio * num_prev), total_indices)
    num_fresh = total_indices - num_reuse

    # Reuse: sample num_reuse indices from prev_unique_indices.
    reuse = prev_unique_indices[torch.randperm(num_prev, device=device)[:num_reuse]]

    # Fresh: sample from [0, index_range) excluding prev_unique_indices.
    if num_fresh > 0 and num_prev < index_range:
        mask = torch.ones(index_range, dtype=torch.bool, device=device)
        mask[prev_unique_indices.long()] = False
        available = mask.nonzero(as_tuple=False).squeeze(1).to(dtype)
        fresh = available[torch.randint(0, len(available), (num_fresh,), device=device)]
    elif num_fresh > 0:
        # index_range exhausted by prev; fall back to sampling from prev.
        fresh = prev_unique_indices[
            torch.randint(0, num_prev, (num_fresh,), device=device)
        ]
    else:
        fresh = torch.empty(0, dtype=dtype, device=device)

    indices = torch.cat([reuse, fresh])
    indices = indices[torch.randperm(total_indices, device=device)]
    return indices, indices.unique()


def _generate_global_kjt_with_overlap(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    overlap_ratio: float,
    prev_unique_per_feature: Dict[str, torch.Tensor],
    device: torch.device,
    pooling_avg: int = 10,
    indices_dtype: torch.dtype = torch.int64,
    lengths_dtype: torch.dtype = torch.int32,
    weighted: bool = False,
    overlap_tables: Optional[Set[str]] = None,
) -> Tuple[KeyedJaggedTensor, Dict[str, torch.Tensor]]:
    """Generates a global KJT with controlled per-feature index overlap vs the prev batch.

    Args:
        tables: embedding table configs defining features and num_embeddings.
        batch_size: global batch size (typically B * W).
        overlap_ratio: target overlap ratio for unique indices across batches.
        prev_unique_per_feature: previous batch's unique indices per feature name.
        pooling_avg: average pooling factor (number of indices per sample).
        device: device for the output tensors.
        indices_dtype: dtype for index tensors.
        lengths_dtype: dtype for length tensors.
        weighted: if True, generate random weights via _assemble_kjt.
        overlap_tables: if provided, only apply overlap control to features of
            tables whose names are in this set; other features use plain random
            indices.

    Returns:
        (global KJT, the new batch's unique indices per overlap feature).
    """
    features: List[str] = []
    lengths_per_feature: List[torch.Tensor] = []
    indices_per_feature: List[torch.Tensor] = []
    new_prev_unique: Dict[str, torch.Tensor] = {}

    for table in tables:
        num_embeddings = table.num_embeddings_post_pruning or table.num_embeddings
        use_overlap = overlap_tables is None or table.name in overlap_tables

        for feature_name in table.feature_names:
            features.append(feature_name)

            lengths = torch.max(
                torch.normal(
                    float(pooling_avg),
                    float(pooling_avg) / 10.0,
                    [batch_size],
                    device=device,
                ),
                torch.tensor(1.0, device=device),
            ).to(lengths_dtype)
            lengths_per_feature.append(lengths)

            total_indices = cast(int, lengths.sum().item())
            if use_overlap:
                prev = prev_unique_per_feature.get(feature_name, None)
                indices, new_unique = _generate_overlap_indices(
                    total_indices=total_indices,
                    index_range=num_embeddings,
                    overlap_ratio=overlap_ratio,
                    prev_unique_indices=prev,
                    device=device,
                    dtype=indices_dtype,
                )
                new_prev_unique[feature_name] = new_unique
            else:
                indices = torch.randint(
                    0,
                    num_embeddings,
                    (total_indices,),
                    device=device,
                    dtype=indices_dtype,
                )
            indices_per_feature.append(indices)

    kjt = ModelInput._assemble_kjt(
        features=features,
        lengths_per_feature=lengths_per_feature,
        indices_per_feature=indices_per_feature,
        weighted=weighted,
        device=device,
    )
    return kjt, new_prev_unique


def _split_kjt(
    global_kjt: KeyedJaggedTensor,
    world_size: int,
) -> List[KeyedJaggedTensor]:
    """Splits a global KJT into world_size per-rank KJTs along the batch dimension.

    Reshapes lengths from [F, B*W] to [F*W, B] and uses a single
    permute_2D_sparse_data call to regroup from feature-major to rank-major
    order, then slices the result.

    Args:
        global_kjt: KJT with stride = B * W.
        world_size: number of ranks to split into.

    Returns:
        A list of world_size KJTs, each with stride = B.
    """
    total_batch = global_kjt.stride()
    local_batch = total_batch // world_size
    keys = global_kjt.keys()
    num_features = len(keys)
    values = global_kjt.values()
    weights = global_kjt.weights_or_none()
    device = values.device

    # Reshape [F, B*W] -> [F*W, B]: each row is one (feature, chunk) pair.
    lengths_fw_b = global_kjt.lengths().view(num_features * world_size, local_batch)

    # Single permutation feature-major -> rank-major: input row (f, r) at index
    # f*W+r, output row (r, f) at index r*F+f, so permute[r*F + f] = f*W + r.
    r_idx = torch.arange(world_size, device=device).unsqueeze(1)
    f_idx = torch.arange(num_features, device=device).unsqueeze(0)
    permute = (f_idx * world_size + r_idx).reshape(-1).int()

    out_lengths, out_values, out_weights = torch.ops.fbgemm.permute_2D_sparse_data(
        permute, lengths_fw_b, values, weights, values.numel()
    )

    # Slice into W rank chunks.
    out_lengths_per_rank = out_lengths.view(world_size, num_features * local_batch)
    value_splits = out_lengths_per_rank.sum(dim=1).tolist()
    value_chunks = out_values.split(value_splits)
    weight_chunks = (
        out_weights.split(value_splits) if weights is not None else [None] * world_size
    )

    return [
        KeyedJaggedTensor(
            keys=list(keys),
            values=value_chunks[r],
            lengths=out_lengths_per_rank[r],
            weights=weight_chunks[r],
        )
        for r in range(world_size)
    ]


def generate_overlapping_batches(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    world_size: int,
    num_batches: int,
    overlap_ratio: float = 0.5,
    weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
    num_float_features: int = 16,
    pooling_avg: int = 10,
    device: Optional[torch.device] = None,
    indices_dtype: torch.dtype = torch.int64,
    lengths_dtype: torch.dtype = torch.int32,
    overlap_tables: Optional[List[str]] = None,
    pin_memory: bool = False,
) -> List[List[ModelInput]]:
    """Generates batches of ModelInput with controlled post-A2A index overlap.

    Intended for PEC RW-sharding training benchmarks/tests. Generates a global
    KJT (batch size B * W) with per-feature overlap control, then splits it into
    per-rank KJTs; under a uniform index distribution, per-rank post-A2A overlap
    approximates the global overlap.

    Args:
        tables: embedding table configs for unweighted (idlist) features.
        batch_size: per-rank batch size (B).
        world_size: number of ranks (W).
        num_batches: number of batches to generate.
        overlap_ratio: target |S_t intersect S_{t+1}| / |S_t| for unique indices
            across consecutive batches (0.0 to 1.0).
        weighted_tables: optional embedding table configs for weighted features.
        num_float_features: number of dense float features.
        pooling_avg: average pooling factor per feature.
        device: device for the output tensors.
        indices_dtype: dtype for index tensors.
        lengths_dtype: dtype for length tensors.
        overlap_tables: if provided, only apply overlap control to tables whose
            names are in this list; other tables use plain random indices.
        pin_memory: if True, pin each ModelInput's tensors for fast async H2D
            copy in the pipeline (matches ModelInput.generate).

    Returns:
        batches[b][r]: ModelInput for batch b, rank r (pre-A2A format).
    """
    device = device if device is not None else torch.device("cpu")
    global_batch_size = batch_size * world_size
    prev_unique_idlist: Dict[str, torch.Tensor] = {}
    prev_unique_idscore: Dict[str, torch.Tensor] = {}
    all_batches: List[List[ModelInput]] = []
    overlap_set: Optional[Set[str]] = (
        set(overlap_tables) if overlap_tables is not None else None
    )

    for _ in range(num_batches):
        global_idlist_kjt, prev_unique_idlist = _generate_global_kjt_with_overlap(
            tables=tables,
            batch_size=global_batch_size,
            overlap_ratio=overlap_ratio,
            prev_unique_per_feature=prev_unique_idlist,
            pooling_avg=pooling_avg,
            device=device,
            indices_dtype=indices_dtype,
            lengths_dtype=lengths_dtype,
            overlap_tables=overlap_set,
        )
        per_rank_idlist = _split_kjt(global_idlist_kjt, world_size)

        per_rank_idscore: Optional[List[KeyedJaggedTensor]] = None
        if weighted_tables:
            global_idscore_kjt, prev_unique_idscore = _generate_global_kjt_with_overlap(
                tables=weighted_tables,
                batch_size=global_batch_size,
                overlap_ratio=overlap_ratio,
                prev_unique_per_feature=prev_unique_idscore,
                pooling_avg=pooling_avg,
                device=device,
                indices_dtype=indices_dtype,
                lengths_dtype=lengths_dtype,
                weighted=True,
                overlap_tables=overlap_set,
            )
            per_rank_idscore = _split_kjt(global_idscore_kjt, world_size)

        batch: List[ModelInput] = []
        for r in range(world_size):
            float_features = torch.rand(batch_size, num_float_features, device=device)
            label = torch.rand(batch_size, device=device)
            idlist = per_rank_idlist[r]
            idscore = per_rank_idscore[r] if per_rank_idscore is not None else None
            if pin_memory:
                # Pin so the pipeline's H2D copy can be truly async, matching the
                # standard ModelInput.generate path.
                float_features, idlist, idscore, label, _ = ModelInput._pin_memory(
                    float_features, idlist, idscore, label
                )
            batch.append(ModelInput(float_features, idlist, idscore, label))
        all_batches.append(batch)

    return all_batches
