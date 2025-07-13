import argparse
import os
from typing import Any, Dict, Tuple

import torch

import torch.nn as nn
import yaml

from .models import make_model_dlrmv2, make_model_dlrmv3


def make_model(
    model_name: str, args: argparse.Namespace, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    if model_name == "dlrmv2":
        # get model configuration from yaml file
        with open(
            os.path.join(os.path.dirname(__file__), "configs", "dlrmv2.yaml"), "r"
        ) as f:
            configs = yaml.safe_load(f)
        # get dataset configuration from yaml file
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "configs",
                f"{args.dataset_name}.yaml",
            ),
            "r",
        ) as f:
            dataset_config = yaml.safe_load(f)
        # combine model and dataset configurations
        configs.update(dataset_config)
        # get the model
        return make_model_dlrmv2(args, configs, device), configs
    elif model_name == "dlrmv3":
        with open(
            os.path.join(os.path.dirname(__file__), "configs", "dlrmv3.yaml"), "r"
        ) as f:
            configs = yaml.safe_load(f)
        # get dataset configuration from yaml file
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "configs",
                f"{args.dataset_name}.yaml",
            ),
            "r",
        ) as f:
            dataset_config = yaml.safe_load(f)
        # combine model and dataset configurations
        configs.update(dataset_config)
        # get the model
        return make_model_dlrmv3(args, configs, device), configs
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
