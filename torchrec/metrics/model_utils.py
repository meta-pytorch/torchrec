#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torchrec.metrics.rec_metric import RecTaskInfo


logger: logging.Logger = logging.getLogger(__name__)


def session_ids_to_tensor(
    session_ids: List[str],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    This function is used to prepare model outputs with session_ids as List[str] to tensor to be consumed by the Metric computation
    """
    curr_id = 1
    session_lengths_list = [0]

    for i, session in enumerate(session_ids[:-1]):
        if session == session_ids[i + 1]:
            session_lengths_list.append(curr_id)
        else:
            session_lengths_list.append(curr_id)
            curr_id += 1

    session_lengths_list.append(curr_id)
    return torch.tensor(session_lengths_list[1:], device=device)


def is_empty_signals(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    return (
        torch.numel(labels) <= 0
        and torch.numel(predictions) <= 0
        and torch.numel(weights) <= 0
    )


def parse_model_outputs(
    label_name: str,
    prediction_name: str,
    weight_name: str,
    model_out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    labels = model_out[label_name].squeeze()
    if not prediction_name:
        assert not weight_name, "weight name must be empty if prediction name is empty"
        return (labels, None, None)
    assert isinstance(labels, torch.Tensor)
    predictions = model_out[prediction_name].squeeze()
    assert isinstance(predictions, torch.Tensor)
    weights = model_out[weight_name].squeeze()
    assert isinstance(weights, torch.Tensor)

    if not is_empty_signals(labels, predictions, weights):
        if labels.dim() == predictions.dim():
            # For vector valued label and prediction we should have shapes
            # labels.size() == (batch_size, dim_vector_valued_label)
            # predictions.size() == (batch_size, dim_vector_valued_prediction)
            # weights.size() == (batch_size,)
            is_vector_valued_label_and_prediction = (
                (labels.dim() == 2)
                and (weights.dim() == 1)
                and (labels.size()[0] == predictions.size()[0])
                and (labels.size()[0] == weights.size()[0])
            )
            if is_vector_valued_label_and_prediction:
                logger.warning(
                    f"""
                    Vector valued labels and predictions are provided. 

                    For vector valued label and prediction we should have shapes 
                    labels.shape: (batch_size, dim_vector_valued_label)
                    predictions.shape: (batch_size, dim_vector_valued_prediction)
                    weights.shape: (batch_size,)

                    The provided labels, predictions and weights comply with the conditions for vector valued labels and predictions. 
                    These conditions are: 
                    1. labels.dim() == 2
                    2. predictions.dim() == 2
                    3. weights.dim() == 1
                    4. labels.size()[0] == predictions.size()[0]
                    5. labels.size()[0] == weights.size()[0]

                    The shapes of labels, predictions and weights are: 
                    labels.shape == {labels.shape}, 
                    predictions.shape == {predictions.shape}, 
                    weights.shape == {weights.shape} 
                    """
                )
            else:
                assert (torch.numel(labels) == torch.numel(predictions)) and (
                    torch.numel(labels) == torch.numel(weights)
                ), (
                    "Expect the same number of elements in labels, predictions, and weights. "
                    f"Instead got {torch.numel(labels)}, {torch.numel(predictions)}, "
                    f"{torch.numel(weights)}"
                )
        else:  # For multiclass models, labels.size() = (batch_size), and predictions.size() = (batch_size, number_of_classes)
            assert torch.numel(labels) == torch.numel(predictions) / predictions.size()[
                -1
            ] and torch.numel(labels) == torch.numel(weights)

        # non-empty tensors need to have rank 1
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
            predictions = predictions.unsqueeze(0)
            weights = weights.unsqueeze(0)

    return labels, predictions, weights


def parse_required_inputs(
    model_out: Dict[str, torch.Tensor],
    required_inputs_list: List[str],
    ndcg_transform_input: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    required_inputs: Dict[str, torch.Tensor] = {}
    for feature in required_inputs_list:
        # convert feature defined from config only
        if ndcg_transform_input:
            model_out[feature] = (
                session_ids_to_tensor(model_out[feature], device=device)
                if isinstance(model_out[feature], list)
                else model_out[feature]
            )
        required_inputs[feature] = model_out[feature].squeeze()
        assert isinstance(required_inputs[feature], torch.Tensor)
    return required_inputs


def parse_task_model_outputs(
    tasks: List[RecTaskInfo],
    model_out: Dict[str, torch.Tensor],
    required_inputs_list: Optional[List[str]] = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    all_labels: Dict[str, torch.Tensor] = {}
    all_predictions: Dict[str, torch.Tensor] = {}
    all_weights: Dict[str, torch.Tensor] = {}
    all_required_inputs: Dict[str, torch.Tensor] = {}
    # Convert session_ids to tensor if NDCG metric
    ndcg_transform_input = False
    for task in tasks:
        labels, predictions, weights = parse_model_outputs(
            task.label_name, task.prediction_name, task.weight_name, model_out
        )
        if predictions is not None and weights is not None:
            if not is_empty_signals(labels, predictions, weights):
                all_labels[task.name] = labels
                all_predictions[task.name] = predictions
                all_weights[task.name] = weights
        else:
            if torch.numel(labels) > 0:
                all_labels[task.name] = labels

        if task.name and task.name.startswith("ndcg"):
            ndcg_transform_input = True

    if required_inputs_list is not None:
        all_required_inputs = parse_required_inputs(
            model_out,
            required_inputs_list,
            ndcg_transform_input,
            device=labels.device,
        )

    return all_labels, all_predictions, all_weights, all_required_inputs
