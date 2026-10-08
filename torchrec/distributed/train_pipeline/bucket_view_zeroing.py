#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Alias-preserving window-start gradient zeroing for gradient accumulation."""

import logging
from typing import Any, Iterable

import torch
from torch.nn.parallel import DistributedDataParallel

logger: logging.Logger = logging.getLogger(__name__)

Target = tuple[torch.nn.Parameter, torch.Tensor]


class BucketViewZeroing:
    """Clear eligible gradients without dropping their DDP bucket aliases.

    When DistributedDataParallel is built with ``gradient_as_bucket_view=True``, each
    ``p.grad`` may reference the reducer bucket. Setting it to ``None`` drops that alias
    and can allocate a second dense gradient during the next backward pass. This class
    selects eligible gradients and zeros them in place.
    """

    def __init__(self) -> None:
        self._warned_skips: set[str] = set()

    def collect(  # noqa: C901 - keep the eligibility checks centralized
        self,
        ddp_modules: Iterable[Any],
        optimizer: torch.optim.Optimizer,
    ) -> list[Target]:
        """Return parameters whose bucket-view gradients can be zeroed in place.

        Excluded parameters retain the default
        ``optimizer.zero_grad(set_to_none=True)`` behavior.
        """
        real_ddps: list[DistributedDataParallel] = [
            m for m in ddp_modules if isinstance(m, DistributedDataParallel)
        ]

        # Use _module_parameters because module.parameters() ignores the DDP ignore list.
        owner_of: dict[int, DistributedDataParallel] = {}
        skipped_ddp_ids: set[int] = set()
        excluded_param_ids: set[int] = set()
        for ddp in real_ddps:
            module_params = getattr(ddp, "_module_parameters", None)
            if module_params is None:
                skipped_ddp_ids.add(id(ddp))
                self._warn_skip(
                    "missing_module_parameters",
                    f"{type(ddp).__name__} does not expose _module_parameters",
                )
                continue
            for p in module_params:
                if not p.requires_grad:
                    continue
                prev = owner_of.get(id(p))
                if prev is not None and prev is not ddp:
                    excluded_param_ids.add(id(p))
                    self._warn_skip(
                        "double_ddp_ownership",
                        f"a parameter of shape {tuple(p.shape)} is reduced by two "
                        "distinct DistributedDataParallel reducers",
                    )
                    continue
                owner_of[id(p)] = ddp

        # Match optimizer.zero_grad by selecting only optimizer-owned parameters.
        owned_param_ids: set[int] = set()
        for group in getattr(optimizer, "param_groups", None) or []:
            if isinstance(group, dict):
                for p in group.get("params", []):
                    owned_param_ids.add(id(p))

        # Select eligible gradients from DDP modules that use bucket views.
        targets: list[Target] = []
        for ddp in real_ddps:
            if id(ddp) in skipped_ddp_ids:
                continue
            if not getattr(ddp, "gradient_as_bucket_view", False):
                continue
            if getattr(ddp, "find_unused_parameters", False):
                self._warn_skip(
                    "find_unused_parameters",
                    f"{type(ddp).__name__} uses find_unused_parameters=True, so its "
                    "used-parameter set is dynamic per micro-batch",
                )
                continue
            for p in ddp._module_parameters:
                if not p.requires_grad:
                    continue
                if id(p) in excluded_param_ids:
                    continue
                if id(p) not in owned_param_ids:
                    continue
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse or grad.layout != torch.strided:
                    continue
                if getattr(grad, "_base", None) is None:
                    self._warn_skip(
                        "standalone_grad",
                        f"a parameter of shape {tuple(p.shape)} on a "
                        "gradient_as_bucket_view=True DDP has a standalone gradient "
                        "(grad._base is None)",
                    )
                    continue
                targets.append((p, grad))
        return targets

    def zero(self, optimizer: torch.optim.Optimizer, targets: list[Target]) -> None:
        """Clear optimizer gradients while preserving selected bucket-view aliases.

        Selected gradients are temporarily removed while ``optimizer.zero_grad()`` clears
        all other parameters. They are then restored and zeroed in place.
        """
        # Always restore aliases if optimizer.zero_grad() raises.
        for p, _grad in targets:
            p.grad = None
        try:
            optimizer.zero_grad(set_to_none=True)
        finally:
            for p, grad in targets:
                p.grad = grad
        # Rebuild groups because a static-graph bucket rebuild can replace the views.
        zero_groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for _p, grad in targets:
            zero_groups.setdefault((grad.device, grad.dtype), []).append(grad)
        for grad_list in zero_groups.values():
            torch._foreach_zero_(grad_list)

    def _warn_skip(self, reason: str, detail: str) -> None:
        """Warn once per reason when a gradient cannot preserve its bucket alias."""
        if reason in self._warned_skips:
            return
        self._warned_skips.add(reason)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return
        logger.warning(
            "[accumulate_into_buckets] using "
            "optimizer.zero_grad(set_to_none=True) for some gradients (%s): %s. "
            "Their dense-gradient memory remains allocated during accumulation.",
            reason,
            detail,
        )
