#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
import torchrec
from torch import nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _GradCheck(torch.autograd.Function):
    """
    Checks if grad has nan. Performs an identity
    during forward. In the backward, we check if grads
    have nan, if not, we return grads as-is, otherwise
    raises an exception
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, tag: str):
        ctx.tag = tag
        return x  # identity

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> Tuple[torch.Tensor, None]:
        check = g.values() if getattr(g, "is_sparse", False) and g.is_sparse else g
        if torch.isnan(check).any() or torch.isinf(check).any():
            raise RuntimeError(f"NaN/Inf detected in gradient entering {ctx.tag}")
        return g, None  # gradient remains unchanged


class DebugEmbeddingCollection(nn.Module):
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
        if self.debug_mode and t.requires_grad:
            return _GradCheck.apply(t, tag)
        return t

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, Any]:
        """
        We obtain EmbeddingCollectionAwaitable object from sharded_model
        In debug mode, we call .wait() on the returned object to get dict
        representation. Then we wrap all the values within GradCheck.
        """
        out = self.ec(features)
        if not self.debug_mode:
            return out

        out = out.wait()
        wrapped: Dict[str, object] = {}
        # we wrap all the values of feature_id with _GradCheck
        for feature_id, jt in out.items():
            wrapped_vals = self._wrap_tensor(
                jt.values(), tag=f"ec[{feature_id}].values"
            )
            wrapped[feature_id] = wrapped_vals
        return wrapped


class DebugEmbeddingBagCollection(nn.Module):
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
        if self.debug_mode and t.requires_grad:
            return _GradCheck.apply(t, tag)
        return t

    def forward(
        self, features: KeyedJaggedTensor
    ) -> torchrec.sparse.jagged_tensor.KeyedTensor:
        """
        We obtain EmbeddingBagCollectionAwaitable object from sharded_model
        In debug mode, we call .wait() on the returned object to get dict
        representation. Then we wrap all the values within GradCheck.
        """
        out = self.ebc(features)
        if not self.debug_mode:
            return out

        if isinstance(
            out, torchrec.distributed.embeddingbag.EmbeddingBagCollectionAwaitable
        ):
            out = out.wait()

        assert isinstance(out, torchrec.sparse.jagged_tensor.KeyedTensor)

        if not isinstance(out, dict):
            out = out.to_dict()

        wrapped: Dict[str, object] = {}
        # we wrap all the values of feature_id with _GradCheck
        for feature_id, jt in out.items():
            wrapped_vals = self._wrap_tensor(jt, tag=f"ebc[{feature_id}].values")
            wrapped[feature_id] = wrapped_vals
        kjt = torchrec.sparse.jagged_tensor.KeyedTensor.from_tensor_list(
            wrapped.keys(), list(wrapped.values())
        )
        return kjt
