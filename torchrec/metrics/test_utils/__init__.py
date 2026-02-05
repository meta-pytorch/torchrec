#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import os
import random
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.auprc import AUPRCMetric
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rauc import RAUCMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo

TestRecMetricOutput = Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]


def gen_test_batch(
    batch_size: int,
    label_name: str = "label",
    prediction_name: str = "prediction",
    weight_name: str = "weight",
    tensor_name: str = "tensor",
    mask_tensor_name: Optional[str] = None,
    label_value: Optional[torch.Tensor] = None,
    prediction_value: Optional[torch.Tensor] = None,
    weight_value: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    n_classes: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    device = torch.device(device or "cpu")
    if seed is not None:
        torch.manual_seed(seed)
    if label_value is not None:
        label = label_value
    else:
        label = torch.randint(0, n_classes or 2, (batch_size,)).double()
    if prediction_value is not None:
        prediction = prediction_value
    else:
        prediction = (
            torch.rand(batch_size, dtype=torch.double)
            if n_classes is None
            else torch.rand(batch_size, n_classes, dtype=torch.double)
        )
    if weight_value is not None:
        weight = weight_value
    else:
        weight = torch.rand(batch_size, dtype=torch.double)
    test_batch = {
        label_name: label.to(device),
        prediction_name: prediction.to(device),
        weight_name: weight.to(device),
        tensor_name: torch.rand(batch_size, dtype=torch.double).to(device),
    }
    if mask_tensor_name is not None:
        if mask is None:
            mask = torch.ones(batch_size, dtype=torch.double).to(device)
        test_batch[mask_tensor_name] = mask

    return test_batch


def gen_test_tasks(
    task_names: List[str],
) -> List[RecTaskInfo]:
    return [
        RecTaskInfo(
            name=task_name,
            label_name=f"{task_name}-label",
            prediction_name=f"{task_name}-prediction",
            weight_name=f"{task_name}-weight",
            tensor_name=f"{task_name}-tensor",
        )
        for task_name in task_names
    ]


def gen_test_timestamps(
    nsteps: int,
) -> List[float]:
    timestamps = [0.0 for _ in range(nsteps)]
    for step in range(1, nsteps):
        time_lapse = random.uniform(1.0, 5.0)
        timestamps[step] = timestamps[step - 1] + time_lapse
    return timestamps


class TestMetric(abc.ABC):
    def __init__(
        self,
        world_size: int,
        rec_tasks: List[RecTaskInfo],
        compute_lifetime_metric: bool = True,
        compute_window_metric: bool = True,
        local_compute_lifetime_metric: bool = True,
        local_compute_window_metric: bool = True,
    ) -> None:
        self.world_size = world_size
        self._rec_tasks = rec_tasks
        self._compute_lifetime_metric = compute_lifetime_metric
        self._compute_window_metric = compute_window_metric
        self._local_compute_lifetime_metric = local_compute_lifetime_metric
        self._local_compute_window_metric = local_compute_window_metric

    @staticmethod
    def _aggregate(
        states: Dict[str, torch.Tensor], new_states: Dict[str, torch.Tensor]
    ) -> None:
        for k, v in new_states.items():
            if k not in states:
                states[k] = torch.zeros_like(v)
            states[k] += v

    @staticmethod
    @abc.abstractmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def compute(
        self,
        model_outs: List[Dict[str, torch.Tensor]],
        nsteps: int,
        batch_window_size: int,
        timestamps: Optional[List[float]],
    ) -> TestRecMetricOutput:
        aggregated_model_out = {}
        lifetime_states, window_states, local_lifetime_states, local_window_states = (
            {task_info.name: {} for task_info in self._rec_tasks} for _ in range(4)
        )
        for i in range(nsteps):
            for k, v in model_outs[i].items():
                aggregated_list = [torch.zeros_like(v) for _ in range(self.world_size)]
                dist.all_gather(aggregated_list, v)
                aggregated_model_out[k] = torch.cat(aggregated_list)
            for task_info in self._rec_tasks:
                states = self._get_states(
                    aggregated_model_out[task_info.label_name],
                    aggregated_model_out[task_info.prediction_name],
                    aggregated_model_out[task_info.weight_name],
                    aggregated_model_out[task_info.tensor_name or "tensor"],
                )
                if self._compute_lifetime_metric:
                    self._aggregate(lifetime_states[task_info.name], states)
                if self._compute_window_metric and nsteps - batch_window_size <= i:
                    self._aggregate(window_states[task_info.name], states)
                local_states = self._get_states(
                    model_outs[i][task_info.label_name],
                    model_outs[i][task_info.prediction_name],
                    model_outs[i][task_info.weight_name],
                    model_outs[i][task_info.tensor_name or "tensor"],
                )
                if self._local_compute_lifetime_metric:
                    self._aggregate(local_lifetime_states[task_info.name], local_states)
                if (
                    self._local_compute_window_metric
                    and nsteps - batch_window_size <= i
                ):
                    self._aggregate(local_window_states[task_info.name], local_states)
        lifetime_metrics = {}
        window_metrics = {}
        local_lifetime_metrics = {}
        local_window_metrics = {}
        for task_info in self._rec_tasks:
            lifetime_metrics[task_info.name] = (
                self._compute(lifetime_states[task_info.name])
                if self._compute_lifetime_metric
                else torch.tensor(0.0)
            )
            window_metrics[task_info.name] = (
                self._compute(window_states[task_info.name])
                if self._compute_window_metric
                else torch.tensor(0.0)
            )
            local_lifetime_metrics[task_info.name] = (
                self._compute(local_lifetime_states[task_info.name])
                if self._local_compute_lifetime_metric
                else torch.tensor(0.0)
            )
            local_window_metrics[task_info.name] = (
                self._compute(local_window_states[task_info.name])
                if self._local_compute_window_metric
                else torch.tensor(0.0)
            )
        return (
            lifetime_metrics,
            window_metrics,
            local_lifetime_metrics,
            local_window_metrics,
        )


BATCH_SIZE = 32
BATCH_WINDOW_SIZE = 5
NSTEPS = 10


def rec_metric_value_test_helper(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Optional[Type[TestMetric]],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    my_rank: int,
    task_names: List[str],
    batch_size: int = BATCH_SIZE,
    nsteps: int = NSTEPS,
    batch_window_size: int = BATCH_WINDOW_SIZE,
    is_time_dependent: bool = False,
    time_dependent_metric: Optional[Dict[Type[RecMetric], str]] = None,
    n_classes: Optional[int] = None,
    zero_weights: bool = False,
    zero_labels: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], ...]]:
    device = torch.device(device or "cpu")
    tasks = gen_test_tasks(task_names)
    model_outs = []
    for _ in range(nsteps):
        weight_value: Optional[torch.Tensor] = None
        if zero_weights:
            weight_value = torch.zeros(batch_size)

        label_value: Optional[torch.Tensor] = None
        if zero_labels:
            label_value = torch.zeros(batch_size)

        _model_outs = [
            gen_test_batch(
                label_name=task.label_name,
                prediction_name=task.prediction_name,
                weight_name=task.weight_name,
                tensor_name=task.tensor_name or "tensor",
                batch_size=batch_size,
                n_classes=n_classes,
                weight_value=weight_value,
                label_value=label_value,
                device=device,
            )
            for task in tasks
        ]
        model_outs.append({k: v for d in _model_outs for k, v in d.items()})

    def get_target_rec_metric_value(
        model_outs: List[Dict[str, torch.Tensor]],
        tasks: List[RecTaskInfo],
        timestamps: Optional[List[float]] = None,
        time_mock: Optional[Mock] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        window_size = world_size * batch_size * batch_window_size
        if n_classes:
            kwargs["number_of_classes"] = n_classes
        if zero_weights:
            kwargs["allow_missing_label_with_zero_weight"] = True

        target_metric_obj = target_clazz(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=target_compute_mode,
            window_size=window_size,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            **kwargs,
        ).to(device)

        for i in range(nsteps):
            # Get required_inputs_list from the target metric
            required_inputs_list = list(target_metric_obj.get_required_inputs())

            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                tasks, model_outs[i], required_inputs_list
            )
            if target_compute_mode in [
                RecComputeMode.FUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            ]:
                labels = torch.stack(list(labels.values()))
                predictions = torch.stack(list(predictions.values()))
                weights = torch.stack(list(weights.values()))

            if timestamps is not None:
                time_mock.return_value = timestamps[i]
            target_metric_obj.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                required_inputs=required_inputs,
            )
        result_metrics = target_metric_obj.compute()
        result_metrics.update(target_metric_obj.local_compute())
        return result_metrics

    def get_test_rec_metric_value(
        model_outs: List[Dict[str, torch.Tensor]],
        tasks: List[RecTaskInfo],
        timestamps: Optional[List[float]] = None,
    ) -> TestRecMetricOutput:
        test_metrics: TestRecMetricOutput = ({}, {}, {}, {})
        if test_clazz is not None:
            test_metric_obj = test_clazz(world_size, tasks)
            test_metrics = test_metric_obj.compute(
                model_outs, nsteps, batch_window_size, timestamps
            )
        return test_metrics

    if is_time_dependent:
        timestamps: Optional[List[float]] = (
            gen_test_timestamps(nsteps) if is_time_dependent else None
        )
        assert time_dependent_metric is not None  # avoid typing issue
        time_dependent_target_clazz_path = time_dependent_metric[target_clazz]
        with patch(time_dependent_target_clazz_path + ".time.monotonic") as time_mock:
            result_metrics = get_target_rec_metric_value(
                model_outs, tasks, timestamps, time_mock, **kwargs
            )
        test_metrics = get_test_rec_metric_value(model_outs, tasks, timestamps)
    else:
        result_metrics = get_target_rec_metric_value(model_outs, tasks, **kwargs)
        test_metrics = get_test_rec_metric_value(model_outs, tasks)

    return result_metrics, test_metrics


def get_launch_config(world_size: int, rdzv_endpoint: str) -> pet.LaunchConfig:
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=world_size,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint=rdzv_endpoint,
        rdzv_configs={"store_type": "file"},
        start_method="spawn",
        monitor_interval=1,
        max_restarts=0,
    )


def rec_metric_gpu_sync_test_launcher(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Optional[Type[TestMetric]],
    metric_name: str,
    task_names: List[str],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    entry_point: Callable[..., None],
    batch_size: int = BATCH_SIZE,
    batch_window_size: int = BATCH_WINDOW_SIZE,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Dict[str, Any],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )

        # launch using torch elastic, launches for each rank
        pet.elastic_launch(lc, entrypoint=entry_point)(
            target_clazz,
            target_compute_mode,
            test_clazz,
            task_names,
            metric_name,
            world_size,
            fused_update_limit,
            compute_on_all_ranks,
            should_validate_update,
            batch_size,
            batch_window_size,
            kwargs.get("n_classes", None),
            False,
            torch.device(device or "cpu"),
        )


def _assert_metric_results(
    test_metrics: TestRecMetricOutput,
    task_names: List[str],
    res: Dict[str, torch.Tensor],
    metric_name: str,
) -> None:
    """Helper function to assert metric results based on metric name."""
    # Serving Calibration uses Calibration naming inconsistently
    if metric_name == "serving_calibration":
        assert torch.allclose(
            test_metrics[1][task_names[0]],
            res[f"{metric_name}-{task_names[0]}|window_calibration"],
        )
    elif metric_name == "effective_sample_rate":
        assert torch.allclose(
            test_metrics[1][task_names[0]],
            res[f"effective_rate-{task_names[0]}|window_{metric_name}"],
        )
    elif metric_name in ("label_average", "prediction_average"):
        # Average metrics use "average" as namespace but different metric names
        assert torch.allclose(
            test_metrics[1][task_names[0]],
            res[f"average-{task_names[0]}|window_{metric_name}"],
        )
    else:
        assert torch.allclose(
            test_metrics[1][task_names[0]],
            res[f"{metric_name}-{task_names[0]}|window_{metric_name}"],
        )


def sync_test_helper(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Optional[Type[TestMetric]],
    task_names: List[str],
    metric_name: str,
    world_size: int,
    fused_update_limit: int = 0,
    compute_on_all_ranks: bool = False,
    should_validate_update: bool = False,
    batch_size: int = BATCH_SIZE,
    batch_window_size: int = BATCH_WINDOW_SIZE,
    n_classes: Optional[int] = None,
    zero_weights: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Dict[str, Any],
) -> None:
    device = torch.device(device or "cpu")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

    tasks = gen_test_tasks(task_names)

    if n_classes:
        kwargs["number_of_classes"] = n_classes

    target_metric_obj = target_clazz(
        world_size=world_size,
        batch_size=batch_size,
        my_rank=rank,
        compute_on_all_ranks=compute_on_all_ranks,
        tasks=tasks,
        window_size=batch_window_size * world_size,
        **kwargs,
    ).to(device)

    weight_value: Optional[torch.Tensor] = None

    _model_outs = [
        gen_test_batch(
            label_name=task.label_name,
            prediction_name=task.prediction_name,
            weight_name=task.weight_name,
            tensor_name=task.tensor_name or "tensor",
            batch_size=batch_size,
            n_classes=n_classes,
            weight_value=weight_value,
            seed=42,  # we set seed because of how test metric places tensors on ranks
            device=device,
        )
        for task in tasks
    ]
    model_outs = []
    model_outs.append({k: v for d in _model_outs for k, v in d.items()})

    # Get required_inputs from the target metric
    required_inputs_list = list(target_metric_obj.get_required_inputs())

    # we send an uneven number of tensors to each rank to test that GPU sync works
    if rank == 0:
        for _ in range(3):
            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                tasks, model_outs[0], required_inputs_list
            )
            target_metric_obj.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                required_inputs=required_inputs,
            )
    elif rank == 1:
        for _ in range(1):
            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                tasks, model_outs[0], required_inputs_list
            )
            target_metric_obj.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                required_inputs=required_inputs,
            )

    # check against test metric
    test_metrics: TestRecMetricOutput = ({}, {}, {}, {})
    if test_clazz is not None:
        test_metric_obj = test_clazz(world_size, tasks)
        # with how testmetric is setup we cannot do asymmertrical updates across ranks
        # so we duplicate model_outs twice to match number of updates in aggregate
        model_outs = model_outs * 2
        test_metrics = test_metric_obj.compute(model_outs, 2, batch_window_size, None)

    res = target_metric_obj.compute()

    if rank == 0:
        _assert_metric_results(test_metrics, task_names, res, metric_name)

    # we also test the case where other rank has more tensors than rank 0
    target_metric_obj.reset()
    if rank == 0:
        for _ in range(1):
            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                tasks, model_outs[0], required_inputs_list
            )
            target_metric_obj.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                required_inputs=required_inputs,
            )
    elif rank == 1:
        for _ in range(3):
            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                tasks, model_outs[0], required_inputs_list
            )
            target_metric_obj.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                required_inputs=required_inputs,
            )

    res = target_metric_obj.compute()

    if rank == 0:
        _assert_metric_results(test_metrics, task_names, res, metric_name)

    dist.destroy_process_group()


def rec_metric_value_test_launcher(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Type[TestMetric],
    metric_name: str,
    task_names: List[str],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    entry_point: Callable[..., None],
    batch_window_size: int = BATCH_WINDOW_SIZE,
    test_nsteps: int = 1,
    n_classes: Optional[int] = None,
    zero_weights: bool = False,
    zero_labels: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )

        # Call the same helper as the actual test to make code coverage visible to
        # the testing system.
        rec_metric_value_test_helper(
            target_clazz,
            target_compute_mode,
            test_clazz=None,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            world_size=1,
            my_rank=0,
            task_names=task_names,
            batch_size=32,
            nsteps=test_nsteps,
            batch_window_size=1,
            n_classes=n_classes,
            zero_weights=zero_weights,
            zero_labels=zero_labels,
            device=device,
            **kwargs,
        )

        pet.elastic_launch(lc, entrypoint=entry_point)(
            target_clazz,
            target_compute_mode,
            task_names,
            test_clazz,
            metric_name,
            fused_update_limit,
            compute_on_all_ranks,
            should_validate_update,
            batch_window_size,
            n_classes,
            test_nsteps,
            zero_weights,
            device,
        )


def rec_metric_accuracy_test_helper(
    world_size: int, entry_point: Callable[..., None]
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )
        pet.elastic_launch(lc, entrypoint=entry_point)()


def metric_test_helper(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    task_names: List[str],
    test_clazz: Type[TestMetric],
    metric_name: str,
    fused_update_limit: int = 0,
    compute_on_all_ranks: bool = False,
    should_validate_update: bool = False,
    batch_window_size: int = BATCH_WINDOW_SIZE,
    n_classes: Optional[int] = None,
    nsteps: int = 1,
    zero_weights: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    is_time_dependent: bool = False,
    time_dependent_metric: Optional[Dict[Type[RecMetric], str]] = None,
    **kwargs: Any,
) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

    target_metrics, test_metrics = rec_metric_value_test_helper(
        target_clazz=target_clazz,
        target_compute_mode=target_compute_mode,
        test_clazz=test_clazz,
        fused_update_limit=fused_update_limit,
        compute_on_all_ranks=False,
        should_validate_update=should_validate_update,
        world_size=world_size,
        my_rank=rank,
        task_names=task_names,
        batch_window_size=batch_window_size,
        n_classes=n_classes,
        nsteps=nsteps,
        is_time_dependent=is_time_dependent,
        time_dependent_metric=time_dependent_metric,
        zero_weights=zero_weights,
        device=device,
        **kwargs,
    )

    if rank == 0:
        for name in task_names:
            # we don't have lifetime metric for AUC due to OOM.
            if (
                target_clazz != AUCMetric
                and target_clazz != AUPRCMetric
                and target_clazz != RAUCMetric
            ):
                assert torch.allclose(
                    target_metrics[
                        f"{str(target_clazz._namespace)}-{name}|lifetime_{metric_name}"
                    ],
                    test_metrics[0][name],
                )
                assert torch.allclose(
                    target_metrics[
                        f"{str(target_clazz._namespace)}-{name}|local_lifetime_{metric_name}"
                    ],
                    test_metrics[2][name],
                )
            assert torch.allclose(
                target_metrics[
                    f"{str(target_clazz._namespace)}-{name}|window_{metric_name}"
                ],
                test_metrics[1][name],
            )

            assert torch.allclose(
                target_metrics[
                    f"{str(target_clazz._namespace)}-{name}|local_window_{metric_name}"
                ],
                test_metrics[3][name],
            )
    dist.destroy_process_group()
