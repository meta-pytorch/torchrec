#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Lightweight EMO decision logger for tracking decisions in TorchRec."""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger: logging.Logger = logging.getLogger(__name__)


class DecisionCategory(Enum):
    """EMO decision category."""

    CLF = "clf"  # Cache Load Factor
    KERNEL = "kernel"
    PROPOSER = "proposer"


@dataclass
class EMODecision:
    """A single EMO decision record."""

    category: DecisionCategory  # e.g., "clf", "kernel", "proposer"
    description: str
    table_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EMOSummary:
    """Accumulated EMO decisions."""

    decisions: List[EMODecision] = field(default_factory=list)

    def add(
        self,
        category: DecisionCategory,
        description: str,
        table_name: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        self.decisions.append(
            EMODecision(
                category=category,
                description=description,
                table_name=table_name,
                metadata=metadata,
            )
        )

    def clear(self) -> None:
        self.decisions.clear()


class EMOLogger:
    """Singleton logger for EMO decisions."""

    _instance: Optional["EMOLogger"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._summary: EMOSummary = EMOSummary()

    @classmethod
    def get_instance(cls) -> "EMOLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def log(
        self,
        category: DecisionCategory,
        level: int,
        description: str,
        table_name: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Log an EMO decision."""
        self._summary.add(category, description, table_name, **metadata)
        logger.log(level, f"[EMO] {category.value}: {description}")

    def get_summary(self) -> EMOSummary:
        return self._summary

    def reset(self) -> None:
        self._summary = EMOSummary()


def get_logger() -> EMOLogger:
    """Get the singleton EMO logger."""
    return EMOLogger.get_instance()


def log_emo_decision(
    category: DecisionCategory,
    level: int,
    description: str,
    table_name: Optional[str] = None,
    **metadata: Any,
) -> None:
    """Log an EMO decision."""
    get_logger().log(category, level, description, table_name, **metadata)
