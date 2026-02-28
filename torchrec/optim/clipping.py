#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from collections import defaultdict
from enum import Enum, unique
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from torchrec.optim.keyed import KeyedOptimizer, OptimizerWrapper

logger: logging.Logger = logging.getLogger()

log_grad_norm: bool = False
use_64bit_grad_norm: bool = False


@unique
class GradientClipping(Enum):
    NORM = "norm"
    VALUE = "value"
    NONE = "none"


class GradientClippingOptimizer(OptimizerWrapper):
    """
    Clips gradients before doing optimization step.

    Args:
        optimizer (KeyedOptimizer): optimizer to wrap
        clipping (GradientClipping): how to clip gradients
        max_gradient (float): max value for clipping
        norm_type (float or str): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        enable_global_grad_clip (bool): whether to enable global gradient clipping.
        param_to_pgs (Dict[torch.nn.Parameter, List[dist.ProcessGroup]], optional): Mapping of parameters
            to process groups. Used for global gradient clipping in n-D model parallelism case.
            Defaults to None, local gradient clipping is used.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        clipping: GradientClipping = GradientClipping.NONE,
        max_gradient: float = 0.1,
        norm_type: Union[float, str] = 2.0,
        enable_global_grad_clip: bool = False,
        param_to_pgs: Optional[
            Dict[torch.nn.Parameter, List[dist.ProcessGroup]]
        ] = None,
    ) -> None:
        super().__init__(optimizer)
        self._clipping = clipping
        self._max_gradient = max_gradient
        self._norm_type = float(norm_type)
        self._check_meta: bool = True
        self._enable_global_grad_clip = enable_global_grad_clip
        self._step_num = 0

        # Group parameters by model parallelism process group if global clipping is enabled.
        # Otherwise, all parameters are treated as replicated and will be clipped locally.
        sharded_param_cnt = 0
        self._replicate_params: List[torch.Tensor] = []
        self._sharded_params: Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]] = (
            defaultdict(list)
        )

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if not self._enable_global_grad_clip:
                    self._replicate_params.append(param)
                    continue
                if param_to_pgs is None or len(param_to_pgs) == 0:
                    self._replicate_params.append(param)
                    continue

                # Group parameters by model parallelism process group.
                if param in param_to_pgs and len(param_to_pgs[param]) != 0:
                    # pyrefly: ignore[bad-index]
                    self._sharded_params[tuple(param_to_pgs[param])].append(param)
                    sharded_param_cnt += 1
                else:
                    self._replicate_params.append(param)
        logger.info(
            f"Optimizer found {sharded_param_cnt} dist params and {len(self._replicate_params)} replicate params."
        )

        # Sanity check: this path is currently not used in any production.
        if self._clipping == GradientClipping.VALUE:
            if sharded_param_cnt > 0:
                raise NotImplementedError(
                    "clip_grad_value_ for sharded parameters is not supported yet"
                )

    def step(self, closure: Any = None) -> None:
        if self._check_meta:
            # skip gradient clipping and early return
            if any(t.device.type == "meta" for t in self._replicate_params):
                super().step(closure)
                return
            if any(
                t.device.type == "meta"
                for params in self._sharded_params.values()
                for t in params
            ):
                super().step(closure)
                return
            self._check_meta = False

        if self._clipping == GradientClipping.NORM:
            # No sharded parameters, local gradient clipping == global gradient clipping
            if len(self._sharded_params) == 0:
                replicate_params = [
                    p._local_tensor if isinstance(p, DTensor) else p
                    for p in self._replicate_params
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters=replicate_params,
                    max_norm=self._max_gradient,
                    norm_type=self._norm_type,
                )
            else:
                self.clip_grad_norm_()

        elif self._clipping == GradientClipping.VALUE:
            torch.nn.utils.clip_grad_value_(
                parameters=self._replicate_params, clip_value=self._max_gradient
            )

        super().step(closure)
        self._step_num += 1

    def clip_grad_norm_(self) -> Optional[Union[float, torch.Tensor]]:
        """Clip the gradient norm of all parameters."""

        # converts self._norm_type to a float if it's a string. Used in the case where self._norm_type is 'inf'.
        all_grads = []
        sharded_params = self._sharded_params
        replicate_params = self._replicate_params

        # Process distributed parameters and gradients
        sharded_grads = {
            pgs: _get_grads(dist_params) for pgs, dist_params in sharded_params.items()
        }

        for grads in sharded_grads.values():
            all_grads.extend(grads)

        # Process replicated parameters and gradients
        replicate_grads = _get_grads(replicate_params)
        all_grads.extend(replicate_grads)

        total_grad_norm = _compute_total_norm(
            replicate_grads=replicate_grads,
            sharded_grads=sharded_grads,
            norm_type=self._norm_type,
            max_grad_norm=self._max_gradient,
        )

        # pyrefly: ignore[redundant-cast]
        clip_coef = cast(torch.Tensor, self._max_gradient / (total_grad_norm + 1e-6))
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        torch._foreach_mul_(all_grads, clip_coef_clamped)
        return total_grad_norm


def _get_grads(
    param_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Get the gradients of a list of parameters. Converts DTensors to local tensors if needed."""
    grads = [
        p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
        for p in param_list
        if p.grad is not None and p.grad.numel() > 0
    ]
    return grads


def _compute_total_norm(
    replicate_grads: List[torch.Tensor],
    sharded_grads: Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]],
    norm_type: float = 2.0,  # can be a normal float, or torch.inf
    max_grad_norm: float = 1.0,
) -> torch.Tensor:
    """
    Given both replicate grads and sharded grads, compute the total norm of the gradients of the full replicate params and the
    full sharded param (parameters with a process group).

    Args:
        replicate_grads (List[torch.Tensor]): list of gradients for replicate params
        sharded_grads (Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]]): dict that maps each process group to a list of gradients for sharded params
        norm_type (float): type of the used p-norm. Can be torch.inf for infinity norm.
        max_grad_norm (float): max gradient norm.
    """

    ## compute the norm |W|^p corresponding to all sharded params W

    combine_norm_operator = torch.maximum if norm_type == torch.inf else torch.add
    initial_value = float("-inf") if norm_type == torch.inf else 0.0
    sharded_grad_norm: torch.Tensor = torch.tensor(initial_value)

    # We need to move sharded_grad_norm to the same device as the first shard so that we can do addition (or take max)
    # this is specifically for the case where sharded_grad_norm is 0, and replicate_grad_norm is not,
    # because by default torch.tensor(0.0) is on cpu, and replicate_grad_norm is on GPU. For MTIA
    # specifically, adding a tensor on cpu and a tensor on GPU will result in an error.
    for pgs, dist_params in sharded_grads.items():
        current_shard_norm = _batch_cal_norm(
            grad_list=dist_params,
            max_norm=max_grad_norm,
            norm_type=norm_type,
            process_groups=pgs,
        )
        sharded_grad_norm = combine_norm_operator(
            sharded_grad_norm.to(current_shard_norm.device, non_blocking=True),
            current_shard_norm,
        )
    # compute |W|^p corresponding to all replicate params W
    # Similar to the case above, we move replicate_grad_norm to the same device as sharded_grad_norm so that we can do addition.
    replicate_grad_norm: torch.Tensor = (
        _batch_cal_norm(
            grad_list=replicate_grads, max_norm=max_grad_norm, norm_type=norm_type
        )
        if replicate_grads
        else torch.tensor(initial_value)
    ).to(sharded_grad_norm.device, non_blocking=True)

    # In the p-norm case, we are given norms |W_sharded|^p and |W_replicate|^p. To compute the total norm, we need to
    # sum them and take the p-th root. In the inf-norm case, we are given max(|W_sharded|) and max(|W_replicate|).
    # To compute the total norm, we need to take max(max(|W_sharded|), max(|W_replicate|).
    combined_norm = combine_norm_operator(replicate_grad_norm, sharded_grad_norm)
    total_grad_norm = (
        combined_norm.pow(1.0 / norm_type) if norm_type != torch.inf else combined_norm
    )

    return total_grad_norm


def _batch_cal_norm(
    grad_list: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    process_groups: Optional[Tuple[dist.ProcessGroup]] = None,
) -> torch.Tensor:
    """Helper function that calculates the p-th power of the norm of a list of gradients in batches.
    If process_groups are passed in, the norm will be aggregated across all ranks in the process group.
    """
    # Filter out any empty tensors for infinity norm.
    # This is necessary because torch._foreach_norm cannot compute infinity norm on empty tensors.
    if norm_type == torch.inf:
        grad_list = [g for g in grad_list if g.numel() > 0]
        if len(grad_list) == 0:
            return torch.tensor(float("-inf"))

    global use_64bit_grad_norm
    if use_64bit_grad_norm:
        grad_norms = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grad_list, norm_type, dtype=torch.float64)),
            norm_type,
        )
    else:
        grad_norms = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grad_list, norm_type)),
            norm_type,
        )

    if norm_type == torch.inf:
        if process_groups is not None:
            for pg in process_groups:
                dist.all_reduce(grad_norms, op=dist.ReduceOp.MAX, group=pg)
    else:
        grad_norms = grad_norms**norm_type
        if process_groups is not None:
            for pg in process_groups:
                dist.all_reduce(grad_norms, group=pg)

    if use_64bit_grad_norm:
        grad_norms = grad_norms.to(torch.float32)

    return grad_norms


def _dedup_to_base_tensors(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    This is a performance optimization specific to FSDP2. Each gradient tensor
    of the same FSDP module share the same base tensor, so for the total norm
    computation, we can directly use the base tensor to reduce the number of
    tensors to compute norm over.
    """
    seen_base_tensors = set()
    base_tensors = []
    for tensor in tensors:
        base_tensor = tensor._base if tensor._base is not None else tensor
        if base_tensor not in seen_base_tensors:
            seen_base_tensors.add(base_tensor)
            base_tensors.append(base_tensor)
    return base_tensors
