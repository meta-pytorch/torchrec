#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import os
from typing import Callable, List, Union

import torch
import torch.nn as nn
import torchrec
import yaml
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.benchmark.benchmark_zch.benchmark_zch_utils import (
    get_module_from_instance,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


def apply_sparse_optimizers(model: nn.Module, args: argparse.Namespace) -> None:
    embedding_optimizer = None
    if args.sparse_optim == "adagrad":
        embedding_optimizer = torch.optim.Adagrad
    elif args.sparse_optim == "sgd":
        embedding_optimizer = torch.optim.SGD
    elif args.sparse_optim == "rowwiseadagrad":
        embedding_optimizer = torchrec.optim.RowWiseAdagrad
    else:
        raise NotImplementedError("Optimizer not supported")
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.sparse_optim == "adagrad":
        optimizer_kwargs["eps"] = args.eps
    elif args.sparse_optim == "rowwiseadagrad":
        optimizer_kwargs["eps"] = args.eps
        optimizer_kwargs["lr"] = args.learning_rate
        optimizer_kwargs["beta1"] = args.beta1
        optimizer_kwargs["beta2"] = args.beta2
        optimizer_kwargs["weight_decay"] = args.weight_decay
    # get model's embedding module's attribute path
    model_config_file_path = os.path.join(
        os.path.dirname(__file__), "configs", f"{args.model_name}.yaml"
    )
    with open(model_config_file_path, "r") as f:
        model_config = yaml.safe_load(f)
    embedding_module_attribute_path = model_config["embedding_module_attribute_path"]
    apply_optimizer_in_backward(
        embedding_optimizer,
        get_module_from_instance(model, embedding_module_attribute_path).parameters(),
        optimizer_kwargs,
    )


def apply_dense_optimizers(
    model: nn.Module, args: argparse.Namespace
) -> KeyedOptimizerWrapper:
    def optimizer_with_params() -> (
        Callable[[List[Union[torch.Tensor, ShardedTensor]]], torch.optim.Optimizer]
    ):
        if args.dense_optim == "adagrad":
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        elif args.dense_optim == "sgd":
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)
        elif args.dense_optim == "adam":
            return lambda params: torch.optim.Adam(
                params, lr=args.learning_rate, eps=args.eps, betas=(0.95, 0.999)
            )
        else:
            raise NotImplementedError("Optimizer not supported")

    # exclude hash identity tables
    dense_optimizer_parameters_dict = dict(
        in_backward_optimizer_filter(model.named_parameters())
    )
    keys_to_remove = []
    for key in dense_optimizer_parameters_dict.keys():
        if "hash_zch_identities" in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del dense_optimizer_parameters_dict[key]

    dense_optimizer = KeyedOptimizerWrapper(
        dense_optimizer_parameters_dict,
        optimizer_with_params(),
    )
    return dense_optimizer


def combine_optimizers(
    sparse_optimizers: Union[KeyedOptimizerWrapper, CombinedOptimizer],
    dense_optimizers: KeyedOptimizerWrapper,
) -> torch.optim.Optimizer:
    return CombinedOptimizer([sparse_optimizers, dense_optimizers])
