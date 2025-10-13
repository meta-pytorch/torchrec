#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional, Type

import torch
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_tower_sharding import (
    EmbeddingTowerCollectionSharder,
    EmbeddingTowerSharder,
)

from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    EmbeddingBagSharder,
)
from torchrec.distributed.fused_embedding import FusedEmbeddingCollectionSharder
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.mc_embedding_modules import (
    BaseManagedCollisionEmbeddingCollectionSharder,
)
from torchrec.distributed.mc_embeddingbag import (
    ShardedManagedCollisionEmbeddingBagCollection,
)
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection


class TestECSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}

        self._sharding_type = sharding_type
        self._kernel_type = kernel_type
        super().__init__(fused_params, qcomm_codecs_registry)

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


class TestEBCSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}

        self._sharding_type = sharding_type
        self._kernel_type = kernel_type
        super().__init__(fused_params, qcomm_codecs_registry)

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


class TestMCSharder(ManagedCollisionCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        self._sharding_type = sharding_type
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class TestEBCSharderMCH(
    BaseManagedCollisionEmbeddingCollectionSharder[
        ManagedCollisionEmbeddingBagCollection
    ]
):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            TestEBCSharder(
                sharding_type, kernel_type, fused_params, qcomm_codecs_registry
            ),
            TestMCSharder(sharding_type, qcomm_codecs_registry),
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    @property
    def module_type(self) -> Type[ManagedCollisionEmbeddingBagCollection]:
        return ManagedCollisionEmbeddingBagCollection

    def shard(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedManagedCollisionEmbeddingBagCollection:
        if device is None:
            device = torch.device("cuda")
        return ShardedManagedCollisionEmbeddingBagCollection(
            module,
            params,
            # pyre-ignore [6]
            ebc_sharder=self._e_sharder,
            mc_sharder=self._mc_sharder,
            env=env,
            device=device,
        )


class TestFusedEBCSharder(FusedEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params={}, qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class TestFusedECSharder(FusedEmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
    ) -> None:
        super().__init__()
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class TestEBSharder(EmbeddingBagSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params


class TestETSharder(EmbeddingTowerSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params


class TestETCSharder(EmbeddingTowerCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params
