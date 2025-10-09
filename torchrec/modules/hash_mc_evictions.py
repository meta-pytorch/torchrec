#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import time
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional, Tuple

import torch
from pyre_extensions import none_throws

from torchrec.sparse.jagged_tensor import JaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


@unique
class HashZchEvictionPolicyName(Enum):
    # eviction based on the time the ID is last seen during training,
    # and a single TTL
    SINGLE_TTL_EVICTION = "SINGLE_TTL_EVICTION"
    # eviction based on the time the ID is last seen during training,
    # and per-feature TTLs
    PER_FEATURE_TTL_EVICTION = "PER_FEATURE_TTL_EVICTION"
    # eviction based on least recently seen ID within the probe range
    LRU_EVICTION = "LRU_EVICTION"


@torch.jit.script
@dataclass
class HashZchEvictionConfig:
    features: List[str]
    single_ttl: Optional[int] = None
    per_feature_ttl: Optional[List[int]] = None


@torch.fx.wrap
def get_kernel_from_policy(
    policy_name: Optional[HashZchEvictionPolicyName],
) -> int:
    return (
        1
        if policy_name is not None
        and policy_name == HashZchEvictionPolicyName.LRU_EVICTION
        else 0
    )


class HashZchEvictionScorer:
    def __init__(self, config: HashZchEvictionConfig) -> None:
        self._config: HashZchEvictionConfig = config

    def gen_score(self, feature: JaggedTensor, device: torch.device) -> torch.Tensor:
        return torch.empty(0, device=device)

    def gen_threshold(self) -> int:
        return -1


class HashZchSingleTtlScorer(HashZchEvictionScorer):
    def gen_score(self, feature: JaggedTensor, device: torch.device) -> torch.Tensor:
        assert (
            self._config.single_ttl is not None and self._config.single_ttl > 0
        ), "To use scorer HashZchSingleTtlScorer, a positive single_ttl is required."

        return torch.full_like(
            feature.values(),
            # pyre-ignore [58]
            self._config.single_ttl + int(time.time() / 3600),
            dtype=torch.int32,
            device=device,
        )

    def gen_threshold(self) -> int:
        return int(time.time() / 3600)


class HashZchPerFeatureTtlScorer(HashZchEvictionScorer):
    def __init__(self, config: HashZchEvictionConfig) -> None:
        super().__init__(config)

        assert self._config.per_feature_ttl is not None and len(
            self._config.features
        ) == len(
            # pyre-ignore [6]
            self._config.per_feature_ttl
        ), "To use scorer HashZchPerFeatureTtlScorer, a 1:1 mapping between features and per_feature_ttl is required."

        self._per_feature_ttl = torch.IntTensor(self._config.per_feature_ttl)

    def gen_score(self, feature: JaggedTensor, device: torch.device) -> torch.Tensor:
        feature_split = feature.weights()
        assert feature_split.size(0) == self._per_feature_ttl.size(0)

        scores = self._per_feature_ttl.repeat_interleave(feature_split) + int(
            time.time() / 3600
        )

        return scores.to(device=device)

    def gen_threshold(self) -> int:
        return int(time.time() / 3600)


@torch.fx.wrap
def get_eviction_scorer(
    policy_name: str, config: HashZchEvictionConfig
) -> HashZchEvictionScorer:
    if policy_name == HashZchEvictionPolicyName.SINGLE_TTL_EVICTION:
        return HashZchSingleTtlScorer(config)
    elif policy_name == HashZchEvictionPolicyName.PER_FEATURE_TTL_EVICTION:
        return HashZchPerFeatureTtlScorer(config)
    elif policy_name == HashZchEvictionPolicyName.LRU_EVICTION:
        return HashZchSingleTtlScorer(config)
    else:
        return HashZchEvictionScorer(config)


class HashZchThresholdEvictionModule(torch.nn.Module):
    """
    This module manages the computation of eviction score for input IDs. Based on the selected
    eviction policy, a scorer is initiated to generate a score for each ID. The kernel
    will use this score to make eviction decisions.

    Args:
        policy_name: an enum value that indicates the eviction policy to use.
        config: a config that contains information needed to run the eviction policy.

    Example::
        module = HashZchThresholdEvictionModule(...)
        score = module(feature)
    """

    _eviction_scorer: HashZchEvictionScorer

    def __init__(
        self,
        policy_name: HashZchEvictionPolicyName,
        config: HashZchEvictionConfig,
    ) -> None:
        super().__init__()

        self._policy_name: HashZchEvictionPolicyName = policy_name
        self._config: HashZchEvictionConfig = config
        self._eviction_scorer = get_eviction_scorer(
            policy_name=self._policy_name,
            config=self._config,
        )

        logger.info(
            f"HashZchThresholdEvictionModule: {self._policy_name=}, {self._config=}"
        )

    def forward(
        self, feature: JaggedTensor, device: torch.device
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            feature: a jagged tensor that contains the input IDs, and their lengths and
                weights (feature split).
            device: device of the tensor.

        Returns:
            a tensor that contains the eviction score for each ID, plus an eviction threshold.
        """
        return (
            self._eviction_scorer.gen_score(feature, device),
            self._eviction_scorer.gen_threshold(),
        )


class HashZchOptEvictionModule(torch.nn.Module):
    """
    This module manages the eviction of IDs from the ZCH table based on the selected eviction policy.
    Args:
        policy_name: an enum value that indicates the eviction policy to use.
    Example:
        module = HashZchOptEvictionModule(policy_name=HashZchEvictionPolicyName.LRU_EVICTION)
    """

    def __init__(
        self,
        policy_name: HashZchEvictionPolicyName,
    ) -> None:
        super().__init__()

        self._policy_name: HashZchEvictionPolicyName = policy_name

    def forward(self, feature: JaggedTensor, device: torch.device) -> Tuple[None, int]:
        """
        Does not apply to this Eviction Policy. Returns None and -1.
        Args:
            feature: No op
        Returns:
            None, -1
        """
        return None, -1


@torch.fx.wrap
def get_eviction_module(
    policy_name: HashZchEvictionPolicyName, config: Optional[HashZchEvictionConfig]
) -> torch.nn.Module:
    if policy_name in (
        HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
        HashZchEvictionPolicyName.PER_FEATURE_TTL_EVICTION,
        HashZchEvictionPolicyName.LRU_EVICTION,
    ):
        return HashZchThresholdEvictionModule(policy_name, none_throws(config))
    else:
        return HashZchOptEvictionModule(policy_name)


class HashZchEvictionModule(torch.nn.Module):
    """
    This module manages the eviction of IDs from the ZCH table based on the selected eviction policy.
    Args:
        policy_name: an enum value that indicates the eviction policy to use.
        device: device of the tensor.
        config: an optional config required if threshold based eviction is selected.
    Example:
        module = HashZchEvictionModule(policy_name=HashZchEvictionPolicyName.LRU_EVICTION)
    """

    def __init__(
        self,
        policy_name: HashZchEvictionPolicyName,
        device: torch.device,
        config: Optional[HashZchEvictionConfig],
    ) -> None:
        super().__init__()

        self._policy_name: HashZchEvictionPolicyName = policy_name
        self._device: torch.device = device
        self._eviction_module: torch.nn.Module = get_eviction_module(
            self._policy_name, config
        )

        logger.info(f"HashZchEvictionModule: {self._policy_name=}, {self._device=}")

    def forward(self, feature: JaggedTensor) -> Tuple[Optional[torch.Tensor], int]:
        """
        Args:
            feature: a jagged tensor that contains the input IDs, and their lengths and
            weights (feature split).

        Returns:
            For threshold eviction, a tensor that contains the eviction score for each ID, plus an eviction threshold. Otherwise None and -1.
        """
        return self._eviction_module(feature, self._device)
