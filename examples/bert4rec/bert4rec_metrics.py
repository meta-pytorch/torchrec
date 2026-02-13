#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List

import torch


def recalls_and_ndcgs_for_ks(
    scores: torch.Tensor, labels: torch.Tensor, ks: List[int]
) -> Dict[str, float]:
    """
    Compute Recalls and NDCGs based

    Args:
        scores (torch.Tensor) the model output tensor containing score of each item
        labels (torch.Tensor): the labels tensor
        ks (List[int]): the metrics we want to validate

    Returns:
        metrics (Dict[str, float]): The performance metrics based on given scores and labels

    """
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    _, cut = torch.sort(-scores, dim=1)
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics["Recall@%d" % k] = (
            (
                hits.sum(1)
                / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
            )
            .mean()
            .cpu()
            .item()
        )

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
            dcg.device
        )
        ndcg = (dcg / idcg).mean()
        metrics["NDCG@%d" % k] = ndcg.cpu().item()

    return metrics
