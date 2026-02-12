#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Backward hook injection utilities for training pipelines.

This module provides utilities for injecting work functions into the backward
pass of EC (EmbeddingCollection) and EBC (EmbeddingBagCollection) modules.
Work functions are registered at specific injection sites and executed during
the backward all-to-all communication phase.

Example usage:
    from torchrec.distributed.train_pipeline.backward_injection import (
        BackwardHookRegistry,
        InjectionSite,
        register_hooks,
    )
    from torchrec.distributed.types import ShardingType

    # Create registry and add hooks
    registry = BackwardHookRegistry()
    registry.add_hook(
        InjectionSite(fqn="sparse_arch.ebc", sharding_type=ShardingType.TABLE_WISE),
        lambda p: p._optimizer.step(),
    )

    # In pipeline progress():
    register_hooks(registry, pipeline, context.output_dist_embeddings_requests)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, TYPE_CHECKING

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.comm_ops import Request
from torchrec.distributed.embedding import EmbeddingCollectionAwaitable
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from torchrec.distributed.types import Awaitable, NoWait, ShardingType


if TYPE_CHECKING:
    from torchrec.distributed.train_pipeline.train_pipelines import (  # @manual  # pyrefly: ignore[missing-import]
        TrainPipeline,
    )


logger: logging.Logger = logging.getLogger(__name__)


# Type alias for work function that receives pipeline reference
BackwardHookWork = Callable[["TrainPipeline"], None]


@dataclass(frozen=True)
class InjectionSite:
    """
    Injection site specification for backward hooks.

    Attributes:
        fqn: Fully qualified name of the module (e.g., "sparse_arch.ebc")
        sharding_type: The sharding type to target (e.g., ShardingType.TABLE_WISE)
    """

    fqn: str
    sharding_type: ShardingType


@dataclass
class BackwardHookRegistry:
    """Registry mapping injection sites to their work functions."""

    hooks: Dict[InjectionSite, List[BackwardHookWork]] = field(default_factory=dict)

    def add_hook(
        self,
        site: InjectionSite,
        work: BackwardHookWork,
    ) -> None:
        """Adds a work function at the specified injection site."""
        if site not in self.hooks:
            self.hooks[site] = []
        self.hooks[site].append(work)

    def work(self, site: InjectionSite) -> List[BackwardHookWork]:
        """Gets all work functions for a site."""
        return self.hooks.get(site, [])


def _filter_awaitables(
    odist_awaitable: Any,
) -> Dict[ShardingType, Awaitable[torch.Tensor]] | None:
    """
    Extracts valid (non-DP) awaitables from an EC/EBC awaitable.

    Args:
        odist_awaitable: The EC/EBC awaitable (may be MC tuple-wrapped)

    Returns:
        Dict mapping ShardingType to awaitable, or None if not a valid
        EC/EBC awaitable. DP sharding (NoWait) awaitables are filtered out.
    """
    # Handle MC EC/EBC tuple wrapping
    if isinstance(odist_awaitable, tuple):
        odist_awaitable = odist_awaitable[0]

    # NOTE: We avoid importing VariableBatchEmbeddingBagCollectionAwaitable directly
    # due to torch.package compatibility issues with repackaging. Instead, we use
    # hasattr to detect EBC-like awaitables (including VB-EBC).
    match odist_awaitable:
        case EmbeddingBagCollectionAwaitable():
            awaitables = odist_awaitable._awaitables
            sharding_types = odist_awaitable._sharding_types
        case EmbeddingCollectionAwaitable():
            awaitables = odist_awaitable._awaitables_per_sharding
            sharding_types = odist_awaitable._sharding_types
        case _ if hasattr(odist_awaitable, "_awaitables") and hasattr(
            odist_awaitable, "_sharding_types"
        ):
            awaitables = odist_awaitable._awaitables
            sharding_types = odist_awaitable._sharding_types
        case _:
            logger.warning(
                f"Unsupported awaitable type: {type(odist_awaitable).__name__}. "
            )
            return None

    # Filter out DP (NoWait) and build dict mapping sharding type to awaitable
    valid_awaitables: Dict[str, Awaitable[torch.Tensor]] = {}
    for w, sharding_type in zip(  # pyrefly: ignore[no-matching-overload]
        awaitables, sharding_types
    ):
        if isinstance(w, NoWait):
            continue

        # pyrefly: ignore[unsupported-operation]
        valid_awaitables[ShardingType(sharding_type)] = w

    return valid_awaitables  # pyrefly: ignore[bad-return]


def _find_awaitable_for_site(
    site: InjectionSite,
    output_dist_embeddings_requests: Dict[str, Any],
) -> Awaitable[torch.Tensor] | None:
    """
    Finds the specific awaitable matching the injection site.

    Args:
        site: The injection site specification
        output_dist_embeddings_requests: Dict mapping FQN to EC/EBC awaitables

    Returns:
        The matching awaitable, or None if not found
    """
    # Find the FQN for this site
    if site.fqn not in output_dist_embeddings_requests:
        logger.warning(f"Could not find module FQN for site: {site}")
        return None

    # Get valid (non-DP) awaitables
    valid_awaitables = _filter_awaitables(output_dist_embeddings_requests[site.fqn])
    if valid_awaitables is None or site.sharding_type not in valid_awaitables:
        logger.warning(
            f"Could not find awaitable for module {site.fqn} "
            f"with sharding type: {site.sharding_type}"
        )
        return None

    return valid_awaitables[site.sharding_type]


def _register_hook_on_tensor(
    awaitable: Awaitable[torch.Tensor],
    hook_fn: Callable[[torch.Tensor], None],
) -> bool:
    """
    Registers a hook on the awaitable's dummy tensor.

    Returns True if successful, False otherwise.
    """
    tensor_awaitable = getattr(awaitable, "_tensor_awaitable", None)
    if tensor_awaitable is None:
        return False

    if isinstance(tensor_awaitable, Request):
        dummy_tensor = tensor_awaitable.dummy_tensor
        dummy_tensor.register_hook(hook_fn)
        return True

    return False


def _create_backward_hook(
    pipeline: "TrainPipeline",
    work_list: List[BackwardHookWork],
    site_name: str,
) -> Callable[[torch.Tensor], None]:
    """
    Creates a backward hook function that executes the given work list.

    Args:
        pipeline: The pipeline instance to pass to work functions
        work_list: List of work functions to execute
        site_name: Name of the injection site (for profiling)

    Returns:
        Hook function to register on a tensor
    """

    def hook_fn(grad: torch.Tensor) -> None:
        with record_function(f"## backward_hook {site_name} ##"):
            for work in work_list:
                work(pipeline)

    return hook_fn


def register_hooks(
    registry: BackwardHookRegistry,
    pipeline: "TrainPipeline",
    output_dist_embeddings_requests: Dict[str, Any],
) -> None:
    """
    Registers all configured backward hooks on output dist tensors.

    This function iterates through all registered hooks in the registry and
    attaches them to the appropriate output dist tensors. The hooks will be
    executed during the backward pass when gradients flow through the tensors.

    Args:
        registry: The backward hook registry containing hook configurations
        pipeline: The pipeline instance to pass to work functions
        output_dist_embeddings_requests: Dict mapping FQN to EC/EBC awaitables
            (typically context.output_dist_embeddings_requests)

    Example:
        register_hooks(
            registry=self._backward_hook_registry,
            pipeline=self,
            output_dist_embeddings_requests=self.contexts[0].output_dist_embeddings_requests,
        )
    """
    if len(registry.hooks) == 0:
        return

    if len(output_dist_embeddings_requests) == 0:
        logger.warning(
            "No output dist requests found. Skipping backward hook registration."
        )
        return

    for site, work_list in registry.hooks.items():
        # Find the specific awaitable matching the site
        awaitable = _find_awaitable_for_site(site, output_dist_embeddings_requests)
        if awaitable is None:
            logger.warning(f"Could not find awaitable for site: {site}")
            continue

        # Register hook on the dummy tensor
        registered = _register_hook_on_tensor(
            awaitable,
            _create_backward_hook(pipeline, work_list, str(site)),
        )

        if registered:
            logger.info(f"Registered backward hook for site: {site} ")
