import argparse
import os

from typing import Union

import yaml
from torch.utils.data import DataLoader

from .preprocess import (
    get_criteo_kaggle_dataloader,
    get_kuairand_1k_dataloader,
    get_kuairand_27k_dataloader,
    get_movielens_1m_dataloader,
)


# from .criteo_kaggle import criteo_kaggle
"""
Get a dataset loader for the zch benchmark
The data in each batch all follows the same format as
- batch
    - dense_features: torch.Tensor or None
    - sparse_features: KeyedJaggedTensor or None
    - labels: torch.Tensor or None
"""


def get_dataloader(
    dataset_name: str,  # the name of the dataset to use
    args: argparse.Namespace,  # the arguments passed to the script
    stage: str = "train",  # the stage of the dataset to use, one of "train", "val", "test"
) -> DataLoader:
    # get the dataset configuration from the yaml file
    assert os.path.exists(
        os.path.join(os.path.dirname(__file__), "configs", f"{dataset_name}.yaml")
    ), f"Dataset {dataset_name} not found"
    with open(
        os.path.join(os.path.dirname(__file__), "configs", f"{dataset_name}.yaml"), "r"
    ) as f:
        dataset_config = yaml.safe_load(f)
    # get the dataset
    if dataset_name == "movielens_1m":
        return get_movielens_1m_dataloader(args, dataset_config, stage)
    elif dataset_name == "criteo_kaggle":
        return get_criteo_kaggle_dataloader(args, dataset_config, stage)
    elif dataset_name == "kuairand_1k":
        return get_kuairand_1k_dataloader(args, dataset_config, stage)
    elif dataset_name == "kuairand_27k":
        return get_kuairand_27k_dataloader(args, dataset_config, stage)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
        return None
