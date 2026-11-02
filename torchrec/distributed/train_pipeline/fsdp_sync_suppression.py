#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Control native FSDP2 reduction and resharding during gradient accumulation."""

from typing import Iterable

import torch
from torch.distributed.fsdp import FSDPModule

# SimpleFSDP manages gradient accumulation separately from native FSDP2. It sets this
# attribute on every module it wraps, nested ones included.
_SIMPLE_FSDP_MARKER: str = "_name_to_simplefsdp_managed_attr_getter"


class FsdpSyncSuppression:
    """Defer native FSDP2 all-reduce and eligible post-backward resharding.

    Reduce-scatter stays enabled while all-reduce is deferred. Restore the flags before
    the next synchronized backward.

    ``set_requires_gradient_sync`` and ``set_is_last_backward`` are never written: the first
    accumulates unsharded gradients, the second skips the reduction completion wait.
    """

    def collect(self, model: torch.nn.Module) -> list[FSDPModule]:
        """Return native FSDP2 modules, including nested modules; exclude SimpleFSDP."""
        # Each module has its own reshard policy, and reshard() is non-recursive.
        targets: list[FSDPModule] = []
        pending: list[torch.nn.Module] = [model]
        while pending:
            module = pending.pop(0)
            if isinstance(module, FSDPModule) and not hasattr(
                module, _SIMPLE_FSDP_MARKER
            ):
                targets.append(module)
            pending.extend(module.children())
        return targets

    def suppress(self, targets: Iterable[FSDPModule]) -> None:
        """Disable all-reduce and eligible post-backward resharding.

        Reduce-scatter stays enabled so accumulation uses sharded gradients.
        """
        for module in targets:
            module.set_requires_all_reduce(False, recurse=False)
            if self._defers_reshard(module):
                module.set_reshard_after_backward(False, recurse=False)

    def restore(self, targets: Iterable[FSDPModule]) -> None:
        """Enable all-reduce and eligible post-backward resharding.

        Call before a synchronized backward so its flags take effect. Changing the flags
        after backward does not reshard its parameters.
        """
        for module in targets:
            module.set_requires_all_reduce(True, recurse=False)
            if self._defers_reshard(module):
                module.set_reshard_after_backward(True, recurse=False)

    def ensure_sharded(self, targets: Iterable[FSDPModule]) -> None:
        """Reshard all target parameter groups before an optimizer step.

        Raise RuntimeError if any group remains unsharded after reshard(). Resharding is
        local, issues no collective, and is a no-op on an already-sharded group.
        """
        for module in targets:
            module.reshard()
            for group in module._get_fsdp_state()._fsdp_param_groups:
                if not group.is_sharded:
                    raise RuntimeError(
                        "FsdpSyncSuppression.ensure_sharded: a parameter group is still "
                        "unsharded after reshard(); the optimizer requires "
                        "sharded parameters."
                    )

    def clear_partial(self, targets: Iterable[FSDPModule]) -> None:
        """Reset root FSDP2 iteration state and discard its gradients.

        reset_iter_state() waits on in-flight collectives and clears all FSDP2 gradient
        state; optimizer.zero_grad() leaves pending reduction buffers intact.
        """
        for module in targets:
            # reset_iter_state() accepts _is_root=None before lazy initialization.
            # Non-root modules are reset through their root.
            if module._get_fsdp_state()._is_root is not False:
                module.reset_iter_state()

    def has_partial(self, targets: Iterable[FSDPModule]) -> bool:
        """Return whether any target has gradients awaiting all-reduce."""
        return any(
            group._partial_reduce_output is not None
            for module in targets
            for group in module._get_fsdp_state()._fsdp_param_groups
        )

    def _defers_reshard(self, module: FSDPModule) -> bool:
        """Return whether post-backward resharding may be deferred.

        Require explicit reshard_after_forward=False for every parameter group. Exclude
        the automatic setting used by default roots.
        """
        state = module._get_fsdp_state()
        if state._auto_reshard_after_forward is not False:
            return False
        groups = state._fsdp_param_groups
        return bool(groups) and all(not g._reshard_after_forward for g in groups)
