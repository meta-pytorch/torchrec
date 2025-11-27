#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
The namespace definition and genration APIs for rec metrics.

The key is defined as the following format:
    f"{namespace}-{task_name}|{metric_key}"
    1. namespace should be one of MetricNamespace.
    2. task_name is user-defined value. If there is no task_name, it should be
       the same as namespace.
    3. metric_key can be:
        a.) One of MetricName.
        b.) One of MetricPrefix + One of MetricName.
        c.) Any string. This arbitrary case should happen only for reporting the
            internal state metrics.
"""

from enum import Enum
from typing import Optional


class StrValueMixin:
    def __str__(self) -> str:
        # pyre-fixme[16]: `StrValueMixin` has no attribute `value`.
        return self.value


class MetricNameBase(StrValueMixin, Enum):
    pass


class MetricName(MetricNameBase):
    DEFAULT = ""

    NE = "ne"
    NE_POSITIVE = "ne_positive"
    SEGMENTED_NE = "segmented_ne"
    LOG_LOSS = "logloss"
    THROUGHPUT = "throughput"
    TOTAL_EXAMPLES = "total_examples"
    ATTEMPT_EXAMPLES = "attempt_examples"
    BATCH_SIZE = "batch_size"
    CTR = "ctr"
    CALIBRATION = "calibration"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R_SQUARED = "r_squared"
    AUC = "auc"
    GAUC = "gauc"
    AUPRC = "auprc"
    RAUC = "rauc"
    GROUPED_AUC = "grouped_auc"
    GROUPED_AUPRC = "grouped_auprc"
    GROUPED_RAUC = "grouped_rauc"
    RECALL_SESSION_LEVEL = "recall_session_level"
    PRECISION_SESSION_LEVEL = "precision_session_level"
    MULTICLASS_RECALL = "multiclass_recall"
    WEIGHTED_AVG = "weighted_avg"
    TOWER_QPS = "qps"
    ACCURACY = "accuracy"
    NDCG = "ndcg"
    XAUC = "xauc"
    SCALAR = "scalar"
    OUTPUT = "output"

    GAUC_NUM_SAMPLES = "gauc_num_samples"
    TOTAL_POSITIVE_EXAMPLES = "total_positive_examples"
    TOTAL_NEGATIVE_EXAMPLES = "total_negative_examples"
    PRECISION = "precision"
    RECALL = "recall"

    SERVING_NE = "serving_ne"
    SERVING_CALIBRATION = "serving_calibration"
    TENSOR_WEIGHTED_AVG = "tensor_weighted_avg"

    CALI_FREE_NE = "cali_free_ne"
    UNWEIGHTED_NE = "unweighted_ne"

    HINDSIGHT_TARGET_PR = "hindsight_target_pr"
    HINDSIGHT_TARGET_PRECISION = "hindsight_target_precision"
    HINDSIGHT_TARGET_RECALL = "hindsight_target_recall"

    EFFECTIVE_SAMPLE_RATE = "effective_sample_rate"

    NMSE = "nmse"
    NRMSE = "nrmse"

    NUM_POSITIVE_SAMPLES = "num_positive_samples"
    SUM_WEIGHTS = "sum_weights"
    NUM_MISSING_LABELS = "num_missing_labels"


class MetricNamespaceBase(StrValueMixin, Enum):
    pass


class MetricNamespace(MetricNamespaceBase):
    DEFAULT = ""

    NE = "ne"
    NE_POSITIVE = "ne_positive"
    SEGMENTED_NE = "segmented_ne"
    RECALIBRATED_NE = "recalibrated_ne"
    RECALIBRATED_CALIBRATION = "recalibrated_calibration"
    THROUGHPUT = "throughput"
    CTR = "ctr"
    CALIBRATION = "calibration"
    MSE = "mse"
    AUC = "auc"
    GAUC = "gauc"
    AUPRC = "auprc"
    RAUC = "rauc"
    MAE = "mae"
    ACCURACY = "accuracy"

    OPTIMIZERS = "optimizers"
    MODEL_CONFIGURATOR = "model_configurator"

    MULTICLASS_RECALL = "multiclass_recall"

    WEIGHTED_AVG = "weighted_avg"
    RECALL_SESSION_LEVEL = "recall_session_level"
    PRECISION_SESSION_LEVEL = "precision_session_level"

    TOWER_QPS = "qps"
    NDCG = "ndcg"
    XAUC = "xauc"

    SCALAR = "scalar"

    PRECISION = "precision"
    RECALL = "recall"

    SERVING_NE = "serving_ne"
    SERVING_CALIBRATION = "serving_calibration"

    OUTPUT = "output"
    TENSOR_WEIGHTED_AVG = "tensor_weighted_avg"

    CALI_FREE_NE = "cali_free_ne"
    UNWEIGHTED_NE = "unweighted_ne"

    HINDSIGHT_TARGET_PR = "hindsight_target_pr"

    # All metrics related to effective rate should be put into this namespace.
    # For example, effective_sample_rate, effective_pos_sample_rate, etc.
    # This is particularly useful for MTML models train with composite pipelines to figure out per-batch blending ratio.
    EFFECTIVE_RATE = "effective_rate"

    NMSE = "nmse"

    NUM_POSITIVE_SAMPLES = "num_positive_samples"
    SUM_WEIGHTS = "sum_weights"
    NUM_MISSING_LABELS = "num_missing_labels"


class MetricPrefix(StrValueMixin, Enum):
    DEFAULT = ""
    LIFETIME = "lifetime_"
    WINDOW = "window_"
    ATTEMPT = "attempt_"


def task_wildcard_metrics_pattern(
    namespace: MetricNamespaceBase,
    metric_name: MetricNameBase,
    metric_prefix: MetricPrefix = MetricPrefix.DEFAULT,
) -> str:
    r"""Get the re (regular expression) pattern to find a set of metrics
    regardless task names. The motivation to have this API is from the past
    bugs which tools hard-code the patterns but the naming change, causing
    some testing issues.
    """
    return rf"{namespace}-.+\|{metric_prefix}{metric_name}"


def compose_metric_namespace(
    namespace: MetricNamespaceBase,
    task_name: str,
) -> str:
    r"""Get the full namespace of a metric based on the input parameters"""
    return f"{namespace}-{task_name}"


def compose_customized_metric_key(
    namespace: str,
    metric_name: str,
    description: Optional[str] = None,
) -> str:
    r"""Get the metric key. The input are unrestricted (string) namespace and
    metric_name. This API should only be used by compose_metric_key() and
    state metrics as the keys of state metrics are unknown.
    """
    return f"{namespace}|{metric_name}{description or ''}"


def compose_metric_key(
    namespace: MetricNamespaceBase,
    task_name: str,
    metric_name: MetricNameBase,
    metric_prefix: MetricPrefix = MetricPrefix.DEFAULT,
    description: Optional[str] = None,
) -> str:
    r"""Get the metric key based on the input parameters"""
    return compose_customized_metric_key(
        compose_metric_namespace(namespace, task_name),
        f"{metric_prefix}{metric_name}",
        description,
    )
