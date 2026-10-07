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

# A parameter and the live bucket-view gradient it is aliased to.
Target = tuple[torch.nn.Parameter, torch.Tensor]


class BucketViewZeroing:
    """Selects bucket-view zeroing targets and applies the alias-preserving clear.

    When DistributedDataParallel is built with ``gradient_as_bucket_view=True``, each
    ``p.grad`` is a view into the reducer's persistent reduction bucket. Clearing with
    ``set_to_none=True`` drops that alias, so the next backward allocates a fresh
    standalone dense gradient and the bucket's memory is held twice for the rest of the
    accumulation window. Zeroing the views in place instead keeps one copy.

    Deciding WHEN a window starts stays with the caller; this object only knows how to
    clear one.
    """

    def __init__(self) -> None:
        self._warned_skips: set[str] = set()

    def collect(  # noqa: C901 — the branches ARE the eligibility contract for in-place bucket-view zeroing; splitting them would scatter a correctness gate
        self,
        ddp_modules: Iterable[Any],
        optimizer: torch.optim.Optimizer,
    ) -> list[Target]:
        """Params (+ their live bucket-view grad) eligible for in-place window-start zero.

        Every exclusion leaves the leaf on the plain ``optimizer.zero_grad(set_to_none=True)``
        path, byte-identical to in-place zeroing being disabled, so degrading is never
        worse than the default.
        """
        real_ddps: list[DistributedDataParallel] = [
            m for m in ddp_modules if isinstance(m, DistributedDataParallel)
        ]

        # Pass 1 (ownership): param-identity -> owning DDP across ALL real DDPs. A grad can
        # alias only one reducer bucket, so a param reduced by two DISTINCT reducers is
        # excluded, as is a DDP with no ``_module_parameters``. Never broaden to
        # ``module.parameters()`` -- it ignores the DDP ignore-list.
        owner_of: dict[int, DistributedDataParallel] = {}
        skipped_ddp_ids: set[int] = set()
        excluded_param_ids: set[int] = set()
        for ddp in real_ddps:
            module_params = getattr(ddp, "_module_parameters", None)
            if module_params is None:
                skipped_ddp_ids.add(id(ddp))
                self._warn_skip(
                    "missing_module_parameters",
                    f"{type(ddp).__name__} exposes no authoritative _module_parameters "
                    "ownership list",
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

        # Ownership intersection: the OFF path clears ONLY the optimizer's own
        # param_groups, so zeroing a DDP-reduced-but-not-owned grad would be a divergence.
        # Duck-typed on ``param_groups`` so a composite optimizer works without importing it.
        owned_param_ids: set[int] = set()
        for group in getattr(optimizer, "param_groups", None) or []:
            if isinstance(group, dict):
                for p in group.get("params", []):
                    owned_param_ids.add(id(p))

        # Pass 2: select in-place-zero targets from the no_sync'd bucket-view DDPs.
        targets: list[Target] = []
        for ddp in real_ddps:
            if id(ddp) in skipped_ddp_ids:
                # Excluded in pass 1 (no ownership list). Already warned.
                continue
            if not getattr(ddp, "gradient_as_bucket_view", False):
                # Grads are not reduction-bucket aliases -> the None path is correct. Must
                # precede the find_unused_parameters warn, which would otherwise claim
                # unreclaimed HBM on a DDP that has none to reclaim.
                continue
            if getattr(ddp, "find_unused_parameters", False):
                self._warn_skip(
                    "find_unused_parameters",
                    f"{type(ddp).__name__} uses find_unused_parameters=True, so its "
                    "used-parameter set is dynamic per micro-batch",
                )
                continue
            # _module_parameters guaranteed present (pass 1).
            for p in ddp._module_parameters:
                if not p.requires_grad:
                    continue
                if id(p) in excluded_param_ids:
                    # Double-owned (pass 1). Already warned.
                    continue
                if id(p) not in owned_param_ids:
                    # DDP-reduced but NOT wrapped-optimizer-owned: OFF never clears
                    # it, so ON must not zero it either -> plain path (leave accumulating).
                    continue
                grad = p.grad
                if grad is None:
                    # Grad-less dense head: stays None (dense optimizer skips it).
                    continue
                if grad.is_sparse or grad.layout != torch.strided:
                    # Not a dense bucket-view alias.
                    continue
                if getattr(grad, "_base", None) is None:
                    # Conservative: the reducer's bucket views are C++-internal, so a
                    # bucket-view is indistinguishable from any other view here.
                    self._warn_skip(
                        "standalone_grad",
                        f"a parameter of shape {tuple(p.shape)} on a "
                        "gradient_as_bucket_view=True DDP holds a STANDALONE grad "
                        "(grad._base is None) -- the bucket-view alias was dropped",
                    )
                    continue
                targets.append((p, grad))
        return targets

    def zero(self, optimizer: torch.optim.Optimizer, targets: list[Target]) -> None:
        """Tree-clear the optimizer while keeping the target grads' bucket-view aliases.

        Targets are hidden (``p.grad = None``), the real optimizer tree-clear runs -- which
        is what propagates fused-embedding LR and applies None semantics to every
        non-target leaf -- then the saved bucket-views are restored and zeroed in place.
        """
        # Restore in a ``finally``: an exception during the tree-clear would otherwise
        # leave targets at None and permanently drop their bucket-view aliases.
        for p, _grad in targets:
            p.grad = None
        try:
            optimizer.zero_grad(set_to_none=True)
        finally:
            for p, grad in targets:
                p.grad = grad
        # Zero the restored aliases in place (view-safe; only reached on success). Grouped
        # writes are storage-safe: DDP reduction-bucket slices share storage but do not
        # overlap. Rebuild the groups from the FRESH ``targets`` every window -- a
        # static_graph bucket rebuild repoints grads to new views.
        zero_groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for _p, grad in targets:
            zero_groups.setdefault((grad.device, grad.dtype), []).append(grad)
        for grad_list in zero_groups.values():
            torch._foreach_zero_(grad_list)

    def _warn_skip(self, reason: str, detail: str) -> None:
        """Warn once per REASON that leaves were excluded from the in-place window-start
        zero and so keep the ``set_to_none`` path.

        Keyed per reason, not per param, so a model with hundreds of affected leaves logs
        once; rank 0 only, because every reason is a property of the rank-uniform DDP
        topology.
        """
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
            "[accumulate_into_buckets] excluding leaves from the alias-preserving "
            "window-start zero (%s): %s. They fall back to "
            "optimizer.zero_grad(set_to_none=True) -- numerically identical to "
            "accumulate_into_buckets=False, but their dense-gradient HBM is NOT "
            "reclaimed across the K-micro window.",
            reason,
            detail,
        )
