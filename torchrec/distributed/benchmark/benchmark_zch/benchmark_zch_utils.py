#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import json
import logging
import os
from typing import Any, Dict, Set

import numpy as np
import torch
import torch.nn as nn
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection


def get_module_from_instance(
    instance: torch.nn.Module, attribute_path: str
) -> nn.Module:
    """
    Dynamically accesses a submodule from an instance.
    Args:
        instance: The instance to start from.
        module_str (str): A string representing the submodule path, e.g., "B.C".
    Returns:
        module: The accessed submodule.
    """
    module_names = attribute_path.split(".")
    module = instance
    for name in module_names:
        module = getattr(module, name)
    return module


def get_logger(log_file_path: str = "") -> logging.Logger:
    """
    Initialize the logger.
    Args:
        log_file_path (str): The path to the log file. If empty, the log will be printed to the
            console.
    Returns:
        logger: The initialized logger.
    """
    # set basic configuration for the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(lineno)d %(module)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %a",
    )
    # set formatter
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(processName)-10s %(levelname)s [%(filename)s %(module)s line: %(lineno)d] %(message)s"
    )
    # create a logger
    logger: logging.Logger = logging.getLogger()
    # append a file handler to the logger if log_file_path is not empty
    if log_file_path:
        # create a file handler
        file_handler = logging.FileHandler(
            log_file_path, mode="w"
        )  # initialize the file handler
        # aet the formatter for the file handler
        file_handler.setFormatter(formatter)
        # add the file handler to the logger
        logger.addHandler(file_handler)  # add file handler to logger
    return logger


class BenchmarkMCProbe(nn.Module):
    """
    The benchmark module for the managed collision embedding collection module
    The benchmark module will record the state of the mcec module before and after the forward pass
    and calculate the statistics of the mcec module
    Args:
        mcec (ManagedCollisionEmbeddingCollection): the managed collision embedding collection module, a dict of {table_name: mcec_module}
        mc_method (str): method for managing collisions, one of ["zch"(sort_zch), "mpzch"]
        rank (int): rank of the current model shard
        log_file_folder (str): folder to store the logging file
    """

    def __init__(
        self,
        mcec: Dict[str, ManagedCollisionEmbeddingCollection],
        mc_method: str,  # method for managing collisions, one of ["zch", "mpzch"]
        rank: int,  # rank of the current model shard
        log_file_folder: str = "benchmark_logs",  # folder to store the logging file
    ) -> None:
        super().__init__()
        # self._mcec is a pointer to the mcec object passed in
        self._mcec = mcec
        # record the mc_method
        self._mc_method = mc_method
        # initialize the logging file handler
        os.makedirs(log_file_folder, exist_ok=True)
        self._log_file_path: str = os.path.join(log_file_folder, f"rank_{rank}.json")
        self._rank = rank  # record the rank of the current model shard
        # # get the output_offsets of the mcec
        # self.per_table_output_offsets = (
        #     {}
        # )  # dict of {table_name [str]: output_offsets [torch.Tensor]} TODO: find out relationship between table_name and feature_name
        # if self._mc_method == "mpzch" or self._mc_method == "":
        #     for table_name, mcec_module in self._mcec.items():
        #         self.per_table_output_offsets[table_name] = (
        #             mcec_module._output_global_offset_tensor
        #         )
        # create a dictionary to store the state of mcec modules
        self.mcec_state: Dict[str, Any] = {}
        # create a dictionary to store the statistics of mch modules
        self._mch_stats: Dict[str, Any] = (
            {}
        )  # dictionary of {table_name [str]: {metric_name [str]: metric_value [int]}}
        self.feature_name_unique_queried_values_set_dict: Dict[str, Set[int]] = {}

    # record mcec state to file
    def record_mcec_state(self, stage: str) -> None:
        """
        record the state of mcec modules to the log file
        The recorded state is a dictionary of
        {{stage: {table_name: {metric_name: state}}}}
        It only covers for the current batch

        params:
            stage (str): before_fwd, after_fwd
        return:
            None
        """
        # check if the stage in the desired options
        assert stage in (
            "before_fwd",
            "after_fwd",
        ), f"stage {stage} is not supported, valid options are before_fwd, after_fwd"
        # create a dictionary to store the state of mcec modules
        if stage not in self.mcec_state:
            self.mcec_state[stage] = {}  # dict of {table_name: {metric_name: state}}
        # if the stage is before_fwd, only record the remapping_table
        # save the mcec table state for each embedding table
        self.mcec_state[stage][
            "table_state"
        ] = {}  # dict of {table_name: {"remapping_table": state}}
        for table_name, mc_module in self._mcec.items():
            self.mcec_state[stage]["table_state"][table_name] = {}
            #
            if self._mc_method == "zch":
                self.mcec_state[stage]["table_state"][table_name][
                    "remapping_table"
                ] = mc_module._mch_sorted_raw_ids
                # save t
            elif self._mc_method == "mpzch" or self._mc_method == "":
                self.mcec_state[stage]["table_state"][table_name]["remapping_table"] = (
                    # pyre-ignore [29] # NOTE: here we did not specify the type of mc_module._hash_zch_identities, but we know it is a parameter of nn.module without gradients
                    mc_module._hash_zch_identities.clone()
                    .to_dense()
                    .squeeze()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            else:
                raise NotImplementedError(
                    f"mc method {self._mc_method} is not supported yet"
                )
        # for before_fwd, we only need to record the remapping_table
        if stage == "before_fwd":
            return
        # for after_fwd, we need to record the feature values
        # check if the "before_fwd" stage is recorded
        assert (
            "before_fwd" in self.mcec_state
        ), "before_fwd stage is not recorded, please call record_mcec_state before calling record_mcec_state after_fwd"
        # create the dirctionary to store the mcec feature values before forward
        self.mcec_state["before_fwd"]["feature_values"] = {}
        # create the dirctionary to store the mcec feature values after forward
        self.mcec_state[stage]["feature_values"] = {}  # dict of {table_name: state}
        # save the mcec feature values for each embedding table
        for table_name, mc_module in self._mcec.items():
            # record the remapped feature values
            if (
                self._mc_method == "mpzch" or self._mc_method == ""
            ):  # when using mpzch mc modules
                # record the remapped feature values first
                self.mcec_state[stage]["feature_values"][table_name] = (
                    # pyre-ignore [29] # NOTE: here we did not specify the type of mc_module.table_name_on_device_remapped_ids_dict[table_name], but we know it is a tensor for remapped ids
                    mc_module.table_name_on_device_remapped_ids_dict[table_name]
                    .cpu()
                    .numpy()
                    .tolist()
                )
                # record the input feature values
                self.mcec_state["before_fwd"]["feature_values"][table_name] = (
                    # pyre-ignore [29] # NOTE: here we did not specify the type of mc_module.table_name_on_device_input_ids_dict[table_name], but we know it is a tensor for input ids
                    mc_module.table_name_on_device_input_ids_dict[table_name]
                    .cpu()
                    .numpy()
                    .tolist()
                )
                # check if the input feature values list is empty
                if (
                    len(self.mcec_state["before_fwd"]["feature_values"][table_name])
                    == 0
                ):
                    # if the input feature values list is empty, make it a list of -2 with the same length as the remapped feature values
                    self.mcec_state["before_fwd"]["feature_values"][table_name] = [
                        -2
                    ] * len(self.mcec_state[stage]["feature_values"][table_name])
            else:  # when using other zch mc modules # TODO: implement the feature value recording for zch
                raise NotImplementedError(
                    f"zc method {self._mc_method} is not supported yet"
                )
        return

    def get_mcec_state(self) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
        """
        get the state of mcec modules
        the state is a dictionary of
        {{stage: {table_name: {data_name: state}}}}
        """
        return self.mcec_state

    def save_mcec_state(self) -> None:
        """
        save the state of mcec modules to the log file
        """
        with open(self._log_file_path, "w") as f:
            json.dump(self.mcec_state, f, indent=4)

    def get_mch_stats(self) -> Dict[str, Dict[str, int]]:
        """
        get the statistics of mch modules
        the statistics is a dictionary of
        {{table_name: {metric_name: metric_value}}}
        """
        return self._mch_stats

    def update(self) -> None:
        """
        Update the ZCH statistics for the current batch
        Params:
            None
        Return:
            None
        Require:
            self.mcec_state is not None and has recorded both "before_fwd" and "after_fwd" for a batch
        Update:
            self._mch_stats
        """
        # create a dictionary to store the statistics for each batch
        batch_stats = (
            {}
        )  # table_name: {hit_cnt: 0, total_cnt: 0, insert_cnt: 0, collision_cnt: 0}
        # calculate the statistics for each rank
        # get the remapping id table before forward pass and the input feature values
        rank_feature_value_before_fwd = self.mcec_state["before_fwd"]["feature_values"]
        # get the remapping id table after forward pass and the remapped feature ids
        rank_feature_value_after_fwd = self.mcec_state["after_fwd"]["feature_values"]
        # for each feature table in the remapped information
        for (
            feature_name,
            remapped_feature_ids,
        ) in rank_feature_value_after_fwd.items():
            # create a new diction for the feature table if not created
            if feature_name not in batch_stats:
                batch_stats[feature_name] = {
                    "hit_cnt": 0,
                    "total_cnt": 0,
                    "insert_cnt": 0,
                    "collision_cnt": 0,
                    "rank_total_cnt": 0,
                    "num_empty_slots": 0,
                    "num_unique_queries": 0,
                }
            # get the input faeture values
            input_feature_values = np.array(rank_feature_value_before_fwd[feature_name])
            # get the values stored in the remapping table for each remapped feature id after forward pass
            prev_remapped_values = np.array(
                self.mcec_state["before_fwd"]["table_state"][f"{feature_name}"][
                    "remapping_table"
                ]
            )[remapped_feature_ids]
            # hit count = number of remapped values that exist in the remapping table before forward pass
            this_rank_hits_count = np.sum(prev_remapped_values == input_feature_values)
            batch_stats[feature_name]["hit_cnt"] += int(this_rank_hits_count)
            # count the number of insertions
            ## insert count = the decreased number of empty slots in the remapping table
            ## before and after forward pass
            num_empty_slots_before = np.sum(
                np.array(
                    self.mcec_state["before_fwd"]["table_state"][f"{feature_name}"][
                        "remapping_table"
                    ]
                )
                == -1
            )
            num_empty_slots_after = np.sum(
                np.array(
                    self.mcec_state["after_fwd"]["table_state"][f"{feature_name}"][
                        "remapping_table"
                    ]
                )
                == -1
            )
            this_rank_insert_count = int(num_empty_slots_before - num_empty_slots_after)
            batch_stats[feature_name]["insert_cnt"] += int(this_rank_insert_count)
            batch_stats[feature_name]["num_empty_slots"] += int(num_empty_slots_after)
            # count the number of total values
            ## total count = the number of remapped values in the remapping table after forward pass
            this_rank_total_count = int(len(remapped_feature_ids))
            # count the number of values redirected to the rank
            batch_stats[feature_name]["rank_total_cnt"] = this_rank_total_count
            batch_stats[feature_name]["total_cnt"] += this_rank_total_count
            # count the number of collisions
            # collision count = total count - hit count - insert count
            this_rank_collision_count = (
                this_rank_total_count - this_rank_hits_count - this_rank_insert_count
            )
            batch_stats[feature_name]["collision_cnt"] += int(this_rank_collision_count)
            # get the unique values in the input feature values
            if feature_name not in self.feature_name_unique_queried_values_set_dict:
                self.feature_name_unique_queried_values_set_dict[feature_name] = set(
                    input_feature_values.tolist()
                )
            else:
                self.feature_name_unique_queried_values_set_dict[feature_name].update(
                    set(input_feature_values.tolist())
                )
            batch_stats[feature_name]["num_unique_queries"] = len(
                self.feature_name_unique_queried_values_set_dict[feature_name]
            )
        self._mch_stats = batch_stats
