#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch
from torch import nn
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class _GradCheck(torch.autograd.Function):
    """
    Checks if grad has nan. Performs an identity
    during forward. In the backward, we check if grads
    have nan, if not, we return grads as-is, otherwise
    raises an exception
    """

    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(ctx, x: torch.Tensor, tag: str):
        ctx.tag = tag
        return x

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, g: torch.Tensor) -> Tuple[torch.Tensor, None]:
        check = g.values() if getattr(g, "is_sparse", False) and g.is_sparse else g
        if torch.isnan(check).any() or torch.isinf(check).any():
            raise RuntimeError(f"NaN/Inf detected in gradient entering {ctx.tag}")
        return g, None


class DebugEmbeddingCollection(nn.Module):
    """
    A debugging wrapper around EmbeddingCollection that can detect NaN/Inf gradients.

    This module wraps the EmbeddingCollection and provides optional gradient checking
    to detect NaN or Inf values during backpropagation. This is useful for debugging
    training issues related to numerical instability.

    Args:
        tables: List of embedding configurations
        device: Device on which to allocate embeddings
        debug_mode: If True, enables gradient checking. If False, operates as normal EmbeddingCollection.

    Example:
        >>> tables = [
        ...     EmbeddingConfig(
        ...         num_embeddings=1000,
        ...         embedding_dim=64,
        ...         name="table_0",
        ...         feature_names=["feature_0"]
        ...     )
        ... ]
        >>> debug_ec = DebugEmbeddingCollection(
        ...     tables=tables,
        ...     device=torch.device("cuda"),
        ...     debug_mode=True
        ... )
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()
        self.ec = EmbeddingCollection(tables=tables, device=device)
        self.debug_mode = debug_mode

    def _wrap_tensor(self, t: torch.Tensor, tag: str) -> torch.Tensor:
        """
        Wraps a tensor with gradient checking if in debug mode.

        Args:
            t: The tensor to wrap
            tag: A tag for identifying the tensor in error messages

        Returns:
            The wrapped tensor if in debug mode and requires_grad=True, otherwise the original tensor
        """
        if self.debug_mode and t.requires_grad:
            return _GradCheck.apply(t, tag)
        else:
            return t

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass through the embedding collection.

        In debug mode, we call .wait() on the returned awaitable object to get dict
        representation, then wrap all tensor values with gradient checking.

        Args:
            features: Input KeyedJaggedTensor

        Returns:
            If debug_mode=False: Returns the original output from EmbeddingCollection
            If debug_mode=True: Returns a dict mapping feature_id to wrapped tensors
        """
        out = self.ec(features)
        if not self.debug_mode:
            return out

        out = out.wait()
        wrapped: Dict[str, JaggedTensor] = {}
        for feature_id, jt in out.items():
            wrapped_vals = self._wrap_tensor(
                jt.values(), tag=f"ec[{feature_id}].values"
            )
            wrapped[feature_id] = JaggedTensor(
                values=wrapped_vals,
                lengths=jt.lengths(),
                weights=jt.weights_or_none(),
            )
        return wrapped


class DebugEmbeddingBagCollection(nn.Module):
    """
    A debugging wrapper around EmbeddingBagCollection that can detect NaN/Inf gradients.

    This module wraps the EmbeddingBagCollection and provides optional gradient checking
    to detect NaN or Inf values during backpropagation. This is useful for debugging
    training issues related to numerical instability in embedding bag operations.

    Args:
        tables: List of embedding bag configurations
        device: Device on which to allocate embeddings
        debug_mode: If True, enables gradient checking. If False, operates as normal EmbeddingBagCollection.

    Example:
        >>> tables = [
        ...     EmbeddingBagConfig(
        ...         name="table_0",
        ...         embedding_dim=64,
        ...         num_embeddings=1000,
        ...         feature_names=["feature_0"],
        ...         pooling=torchrec.PoolingType.SUM
        ...     )
        ... ]
        >>> debug_ebc = DebugEmbeddingBagCollection(
        ...     tables=tables,
        ...     device=torch.device("cuda"),
        ...     debug_mode=True
        ... )
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        device: torch.device,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(tables=tables, device=device)
        self.embedding_bag_configs = self.ebc.embedding_bag_configs
        self.debug_mode = debug_mode

    def _wrap_tensor(self, t: torch.Tensor, tag: str) -> torch.Tensor:
        """
        Wraps a tensor with gradient checking if in debug mode.

        Args:
            t: The tensor to wrap
            tag: A tag for identifying the tensor in error messages

        Returns:
            The wrapped tensor if in debug mode and requires_grad=True, otherwise the original tensor
        """
        if self.debug_mode and t.requires_grad:
            return _GradCheck.apply(t, tag)
        else:
            return t

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Forward pass through the embedding bag collection.

        In debug mode, we call .wait() on the returned awaitable object to get KeyedTensor,
        convert it to dict, wrap all tensor values with gradient checking, and reconstruct KeyedTensor.

        Args:
            features: Input KeyedJaggedTensor

        Returns:
            KeyedTensor with embeddings (wrapped with gradient checking if debug_mode=True)
        """
        out = self.ebc(features)
        if not self.debug_mode:
            return out

        if isinstance(out, EmbeddingBagCollectionAwaitable):
            out = out.wait()

        assert isinstance(out, KeyedTensor), f"Expected KeyedTensor, got {type(out)}"

        out_dict = out.to_dict() if not isinstance(out, dict) else out

        wrapped: Dict[str, torch.Tensor] = {}
        for feature_id, tensor_val in out_dict.items():
            wrapped_vals = self._wrap_tensor(
                tensor_val, tag=f"ebc[{feature_id}].values"
            )
            wrapped[feature_id] = wrapped_vals
        kt = KeyedTensor.from_tensor_list(list(wrapped.keys()), list(wrapped.values()))
        return kt
