#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import abc
from logging import getLogger, Logger
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from torch import nn
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


logger: Logger = getLogger(__name__)


@torch.fx.wrap
def apply_mc_method_to_jt_dict(
    mc_module: nn.Module,
    method: str,
    features_dict: Dict[str, JaggedTensor],
) -> Dict[str, JaggedTensor]:
    """
    Applies an MC method to a dictionary of JaggedTensors, returning the updated dictionary with same ordering
    """
    attr = getattr(mc_module, method)
    return attr(features_dict)


@torch.fx.wrap
def _update(
    base: Optional[Dict[str, JaggedTensor]], delta: Dict[str, JaggedTensor]
) -> Dict[str, JaggedTensor]:
    res: Dict[str, JaggedTensor] = {}
    if base is not None:
        for k, v in base.items():
            res[k] = v
    for k, v in delta.items():
        res[k] = v
    return res


@torch.fx.wrap
def _cat_jagged_values(jd: Dict[str, JaggedTensor]) -> torch.Tensor:
    return torch.cat([jt.values() for jt in jd.values()])


@torch.fx.wrap
def _cat_jagged_lengths(jd: Dict[str, JaggedTensor]) -> torch.Tensor:
    return torch.cat([jt.lengths() for jt in jd.values()])


# TODO: keep the old implementation for backward compatibility and will remove it later
@torch.fx.wrap
def _mcc_lazy_init(
    features: KeyedJaggedTensor,
    feature_names: List[str],
    features_order: List[int],
    created_feature_order: bool,
) -> Tuple[KeyedJaggedTensor, bool, List[int]]:  # features_order
    input_feature_names: List[str] = features.keys()
    if not created_feature_order:
        for f in feature_names:
            features_order.append(input_feature_names.index(f))

        if features_order == list(range(len(input_feature_names))):
            features_order = torch.jit.annotate(List[int], [])
        created_feature_order = True

    if len(features_order) > 0:
        features = features.permute(
            features_order,
        )

    return (features, created_feature_order, features_order)


@torch.fx.wrap
def _mcc_lazy_init_inplace(
    features: KeyedJaggedTensor,
    feature_names: List[str],
    features_order: List[int],
    created_feature_order: List[bool],
) -> KeyedJaggedTensor:
    input_feature_names: List[str] = features.keys()
    if not created_feature_order or not created_feature_order[0]:
        for f in feature_names:
            features_order.append(input_feature_names.index(f))

        if features_order == list(range(len(input_feature_names))):
            features_order.clear()

        if len(created_feature_order) > 0:
            created_feature_order[0] = True
        else:
            created_feature_order.append(True)

    if len(features_order) > 0:
        features = features.permute(
            features_order,
        )

    return features


@torch.fx.wrap
def _get_length_per_key(kjt: KeyedJaggedTensor) -> torch.Tensor:
    return torch.tensor(kjt.length_per_key())


@torch.no_grad()
def dynamic_threshold_filter(
    id_counts: torch.Tensor,
    threshold_skew_multiplier: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Threshold is total_count / num_ids * threshold_skew_multiplier. An id is
    added if its count is strictly greater than the threshold.
    """

    num_ids = id_counts.numel()
    total_count = id_counts.sum()

    BASE_THRESHOLD = 1 / num_ids
    threshold_mass = BASE_THRESHOLD * threshold_skew_multiplier

    threshold = threshold_mass * total_count
    threshold_mask = id_counts > threshold

    return threshold_mask, threshold


@torch.no_grad()
def average_threshold_filter(
    id_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Threshold is average of id_counts. An id is added if its count is strictly
    greater than the mean.
    """
    if id_counts.dtype != torch.float:
        id_counts = id_counts.float()
    threshold = id_counts.mean()
    threshold_mask = id_counts > threshold

    return threshold_mask, threshold


@torch.no_grad()
def probabilistic_threshold_filter(
    id_counts: torch.Tensor,
    per_id_probability: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Each id has probability per_id_probability of being added. For example,
    if per_id_probability is 0.01 and an id appears 100 times, then it has a 60%
    of being added. More precisely, the id score is 1 - (1 - per_id_probability) ^ id_count,
    and for a randomly generated threshold, the id score is the chance of it being added.
    """
    probability = torch.full_like(id_counts, 1 - per_id_probability, dtype=torch.float)
    id_scores = 1 - torch.pow(probability, id_counts)

    threshold: torch.Tensor = torch.rand(id_counts.size(), device=id_counts.device)
    threshold_mask = id_scores > threshold

    return threshold_mask, threshold


class ManagedCollisionModule(nn.Module):
    """
    Abstract base class for ManagedCollisionModule.
    Maps input ids to range [0, max_output_id).

    Args:
        max_output_id (int): Max output value of remapped ids.
        input_hash_size (int): Max value of input range i.e. [0, input_hash_size)
        remapping_range_start_index (int): Relative start index of remapping range
        device (torch.device): default compute device.

    Example::
        jt = JaggedTensor(...)
        mcm = ManagedCollisionModule(...)
        mcm_jt = mcm(fp)
    """

    def __init__(
        self,
        device: torch.device,
        output_segments: List[int],
        skip_state_validation: bool = False,
    ) -> None:
        super().__init__()
        self._device = device

        if skip_state_validation:
            logger.warning(
                "Skipping validation on ManagedCollisionModule.  This module may not be Reshard-able as a result"
            )
            return

        # limited to max of 1024 RW shards
        assert (
            len(output_segments) <= 1025
        ), "ManagedCollisionModule limited to 1024 shards"

        self.register_buffer(
            "_output_segments_tensor",
            torch.tensor(
                output_segments + [-1] * (1025 - len(output_segments)),
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_current_iter_tensor",
            torch.tensor(
                [0],
                dtype=torch.int64,
                device=self.device,
            ),
        )

        def _load_state_dict_post_hook(
            module: "ManagedCollisionModule",
            incompatible_keys: torch.nn.modules.module._IncompatibleKeys,
        ) -> None:
            module.validate_state()

        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)

    @abc.abstractmethod
    def preprocess(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @abc.abstractmethod
    def evict(self) -> Optional[torch.Tensor]:
        """
        Returns None if no eviction should be done this iteration. Otherwise, return ids of slots to reset.
        On eviction, this module should reset its state for those slots, with the assumptionn that the downstream module
        will handle this properly.
        """
        pass

    @abc.abstractmethod
    def remap(self, features: Dict[str, JaggedTensor]) -> Dict[str, JaggedTensor]:
        pass

    @abc.abstractmethod
    def profile(self, features: Dict[str, JaggedTensor]) -> Dict[str, JaggedTensor]:
        pass

    @abc.abstractmethod
    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        pass

    @abc.abstractmethod
    def output_size(self) -> int:
        """
        Returns numerical range of output, for validation vs. downstream embedding lookups
        """
        pass

    @abc.abstractmethod
    def input_size(self) -> int:
        """
        Returns numerical range of input, for sharding info
        """
        pass

    @abc.abstractmethod
    def buckets(self) -> int:
        """
        Returns number of uniform buckets, relevant to resharding
        """
        pass

    @abc.abstractmethod
    def validate_state(self) -> None:
        """
        Validates that the state of the module after loading from checkpoint
        """
        pass

    @abc.abstractmethod
    def open_slots(self) -> torch.Tensor:
        """
        Returns number of unused slots in managed collision module
        """
        pass

    @abc.abstractmethod
    def rebuild_with_output_id_range(
        self,
        output_id_range: Tuple[int, int],
        output_segments: List[int],
        device: Optional[torch.device] = None,
    ) -> "ManagedCollisionModule":
        """
        Used for creating local MC modules for RW sharding
        """
        pass


class ManagedCollisionCollection(nn.Module):
    """
    ManagedCollisionCollection represents a collection of managed collision modules.
    The inputs passed to the MCC will be remapped by the managed collision modules
        and returned.
    Args:
        managed_collision_modules (Dict[str, ManagedCollisionModule]): Dict of managed collision modules
        embedding_confgs (List[BaseEmbeddingConfig]): List of embedding configs, for each table with a managed collsion module
    """

    _table_to_features: Dict[str, List[str]]
    _features_order: List[int]
    _created_feature_order: List[bool]  # use list for inplace update in leaf function

    def __init__(
        self,
        managed_collision_modules: Dict[str, ManagedCollisionModule],
        embedding_configs: Sequence[BaseEmbeddingConfig],
        need_preprocess: bool = True,
    ) -> None:
        super().__init__()
        self._managed_collision_modules = nn.ModuleDict(managed_collision_modules)
        self._embedding_configs = embedding_configs
        self.need_preprocess = need_preprocess
        self._feature_to_table: Dict[str, str] = {
            feature: config.name
            for config in embedding_configs
            for feature in config.feature_names
        }
        self._table_to_features: Dict[str, List[str]] = {
            config.name: config.feature_names for config in embedding_configs
        }

        self._table_feature_splits: List[int] = [
            len(features) for features in self._table_to_features.values()
        ]

        table_to_config = {config.name: config for config in embedding_configs}

        for name, config in table_to_config.items():
            if name not in managed_collision_modules:
                raise ValueError(
                    f"Table {name} is not present in managed_collision_modules"
                )
            assert (
                managed_collision_modules[name].output_size() == config.num_embeddings
            ), (
                f"max_output_id in managed collision module for {name} "
                f"must match {config.num_embeddings}"
            )
        self._feature_names: List[str] = [
            feature for config in embedding_configs for feature in config.feature_names
        ]
        self._created_feature_order: List[bool] = [False]
        self._features_order = []

    def embedding_configs(self) -> Sequence[BaseEmbeddingConfig]:
        return self._embedding_configs

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        features = _mcc_lazy_init_inplace(
            features,
            self._feature_names,
            self._features_order,
            self._created_feature_order,
        )

        feature_splits: List[KeyedJaggedTensor] = features.split(
            self._table_feature_splits
        )

        output: Optional[Dict[str, JaggedTensor]] = None
        for i, (table, mc_module) in enumerate(self._managed_collision_modules.items()):
            kjt: KeyedJaggedTensor = feature_splits[i]
            mc_input: Dict[str, JaggedTensor] = {
                table: JaggedTensor(
                    values=kjt.values(),
                    lengths=kjt.lengths(),
                    weights=_get_length_per_key(kjt),
                )
            }
            mc_input = mc_module(mc_input)
            output = _update(output, mc_input)

        assert output is not None
        values: torch.Tensor = _cat_jagged_values(output)
        lengths: torch.Tensor = _cat_jagged_lengths(output)
        return KeyedJaggedTensor(
            keys=features.keys(),
            values=values,
            lengths=lengths,
            weights=features.weights_or_none(),
        )

    def evict(self) -> Dict[str, Optional[torch.Tensor]]:
        evictions: Dict[str, Optional[torch.Tensor]] = {}
        for (
            table,
            managed_collision_module,
        ) in self._managed_collision_modules.items():
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            evictions[table] = managed_collision_module.evict()
        return evictions

    def open_slots(self) -> Dict[str, torch.Tensor]:
        open_slots: Dict[str, torch.Tensor] = {}
        for (
            table,
            managed_collision_module,
        ) in self._managed_collision_modules.items():
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            open_slots[table] = managed_collision_module.open_slots()
        return open_slots


class MCHEvictionPolicyMetadataInfo(NamedTuple):
    metadata_name: str
    is_mch_metadata: bool
    is_history_metadata: bool


class MCHEvictionPolicy(abc.ABC):
    def __init__(
        self,
        metadata_info: List[MCHEvictionPolicyMetadataInfo],
        threshold_filtering_func: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Union[float, torch.Tensor]]]
        ] = None,  # experimental
    ) -> None:
        """
        threshold_filtering_func (Optional[Callable]): function used to filter incoming ids before update/eviction. experimental feature.
            [input: Tensor] the function takes as input a 1-d tensor of unique id counts.
            [output1: Tensor] the function returns a boolean_mask or index array of corresponding elements in the input tensor that pass the filter.
            [output2: float, Tensor] the function returns the threshold that will be used to filter ids before update/eviction. all values <= this value will be filtered out.

        """
        self._metadata_info = metadata_info
        self._threshold_filtering_func = threshold_filtering_func

    @property
    @abc.abstractmethod
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        pass

    @abc.abstractmethod
    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        """
        Args:
        current_iter (int): current iteration
        incoming_ids (torch.Tensor): incoming ids
        history_metadata (Dict[str, torch.Tensor]): history metadata dict

        Compute and record metadata based on incoming ids
            for the implemented eviction policy.
        """
        pass

    @abc.abstractmethod
    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
        threshold_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
        history_metadata (Dict[str, torch.Tensor]): history metadata dict
        additional_ids (torch.Tensor): additional ids to be used as part of history
        unique_inverse_mapping (torch.Tensor): torch.unique inverse mapping generated from
            torch.cat[history_accumulator, additional_ids]. used to map history metadata tensor
            indices to their coalesced tensor indices.

        Coalesce metadata history buffers and return dict of processed metadata tensors.
        """
        pass

    @abc.abstractmethod
    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:


        Returns Tuple of (evicted_indices, selected_new_indices) where:
            evicted_indices are indices in the mch map to be evicted, and
            selected_new_indices are the indices of the ids in the coalesced
            history that are to be added to the mch.
        """
        pass

    def _compute_selected_eviction_and_replacement_indices(
        self,
        pivot: int,
        eviction_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE these are like indices
        argsorted_eviction_scores = torch.argsort(
            eviction_scores, descending=True, stable=True
        )

        # indices with values >= zch_size in the top zch_size scores correspond
        #   to new incoming ids to be added to zch
        selected_new_ids_mask = argsorted_eviction_scores[:pivot] >= pivot
        # indices with values < zch_size outside the top zch_size scores correspond
        #   to existing zch ids to be evicted
        evicted_ids_mask = argsorted_eviction_scores[pivot:] < pivot
        evicted_indices = argsorted_eviction_scores[pivot:][evicted_ids_mask]
        selected_new_indices = (
            argsorted_eviction_scores[:pivot][selected_new_ids_mask] - pivot
        )

        return evicted_indices, selected_new_indices


class LFU_EvictionPolicy(MCHEvictionPolicy):
    def __init__(
        self,
        threshold_filtering_func: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Union[float, torch.Tensor]]]
        ] = None,  # experimental
    ) -> None:
        super().__init__(
            metadata_info=[
                MCHEvictionPolicyMetadataInfo(
                    metadata_name="counts",
                    is_mch_metadata=True,
                    is_history_metadata=False,
                ),
            ],
            threshold_filtering_func=threshold_filtering_func,
        )

    @property
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        return self._metadata_info

    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        # no-op; no history buffers
        pass

    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
        threshold_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # no-op; no history buffers
        return {}

    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mch_counts = mch_metadata["counts"]
        # update metadata for matching ids
        mch_counts[
            coalesced_history_mch_matching_indices
        ] += coalesced_history_sorted_unique_ids_counts[
            coalesced_history_mch_matching_elements_mask
        ]

        # incoming non-matching ids
        new_sorted_uniq_ids_counts = coalesced_history_sorted_unique_ids_counts[
            ~coalesced_history_mch_matching_elements_mask
        ]

        # TODO: find cleaner way to avoid last element of zch

        mch_counts[mch_size - 1] = torch.iinfo(torch.int64).max

        merged_counts = torch.cat(
            [
                mch_counts,
                new_sorted_uniq_ids_counts,
            ]
        )
        # calculate evicted and replacement indices
        (
            evicted_indices,
            selected_new_indices,
        ) = self._compute_selected_eviction_and_replacement_indices(
            mch_size,
            merged_counts,
        )

        # update metadata for evicted ids
        mch_counts[evicted_indices] = new_sorted_uniq_ids_counts[selected_new_indices]

        return evicted_indices, selected_new_indices


class LRU_EvictionPolicy(MCHEvictionPolicy):
    def __init__(
        self,
        decay_exponent: float = 1.0,
        threshold_filtering_func: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Union[float, torch.Tensor]]]
        ] = None,  # experimental
    ) -> None:
        super().__init__(
            metadata_info=[
                MCHEvictionPolicyMetadataInfo(
                    metadata_name="last_access_iter",
                    is_mch_metadata=True,
                    is_history_metadata=True,
                ),
            ],
            threshold_filtering_func=threshold_filtering_func,
        )
        self._decay_exponent = decay_exponent

    @property
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        return self._metadata_info

    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        history_last_access_iter = history_metadata["last_access_iter"]
        history_last_access_iter[:] = current_iter

    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
        threshold_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        coalesced_history_metadata: Dict[str, torch.Tensor] = {}
        history_last_access_iter = history_metadata["last_access_iter"]
        if additional_ids is not None:
            history_last_access_iter = torch.cat(
                [
                    history_last_access_iter,
                    torch.full_like(additional_ids, current_iter),
                ]
            )
        coalesced_history_metadata["last_access_iter"] = torch.zeros_like(
            unique_ids_counts
        ).scatter_reduce_(
            0,
            unique_inverse_mapping,
            history_last_access_iter,
            reduce="amax",
            include_self=False,
        )
        if threshold_mask is not None:
            coalesced_history_metadata["last_access_iter"] = coalesced_history_metadata[
                "last_access_iter"
            ][threshold_mask]
        return coalesced_history_metadata

    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mch_last_access_iter = mch_metadata["last_access_iter"]

        # sort coalesced history metadata
        coalesced_history_metadata["last_access_iter"].copy_(
            coalesced_history_metadata["last_access_iter"][
                coalesced_history_argsort_mapping
            ]
        )
        coalesced_history_sorted_uniq_ids_last_access_iter = coalesced_history_metadata[
            "last_access_iter"
        ]

        # update metadata for matching ids
        mch_last_access_iter[coalesced_history_mch_matching_indices] = (
            coalesced_history_sorted_uniq_ids_last_access_iter[
                coalesced_history_mch_matching_elements_mask
            ]
        )

        # incoming non-matching ids
        new_sorted_uniq_ids_last_access = (
            coalesced_history_sorted_uniq_ids_last_access_iter[
                ~coalesced_history_mch_matching_elements_mask
            ]
        )

        # TODO: find cleaner way to avoid last element of zch
        mch_last_access_iter[mch_size - 1] = current_iter
        merged_access_iter = torch.cat(
            [
                mch_last_access_iter,
                new_sorted_uniq_ids_last_access,
            ]
        )
        # lower scores are evicted first.
        merged_eviction_scores = torch.neg(
            torch.pow(
                current_iter - merged_access_iter + 1,
                self._decay_exponent,
            )
        )

        # calculate evicted and replacement indices
        (
            evicted_indices,
            selected_new_indices,
        ) = self._compute_selected_eviction_and_replacement_indices(
            mch_size,
            merged_eviction_scores,
        )

        mch_last_access_iter[evicted_indices] = new_sorted_uniq_ids_last_access[
            selected_new_indices
        ]

        return evicted_indices, selected_new_indices


class DistanceLFU_EvictionPolicy(MCHEvictionPolicy):
    def __init__(
        self,
        decay_exponent: float = 1.0,
        threshold_filtering_func: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Union[float, torch.Tensor]]]
        ] = None,  # experimental
    ) -> None:
        super().__init__(
            metadata_info=[
                MCHEvictionPolicyMetadataInfo(
                    metadata_name="counts",
                    is_mch_metadata=True,
                    is_history_metadata=False,
                ),
                MCHEvictionPolicyMetadataInfo(
                    metadata_name="last_access_iter",
                    is_mch_metadata=True,
                    is_history_metadata=True,
                ),
            ],
            threshold_filtering_func=threshold_filtering_func,
        )
        self._decay_exponent = decay_exponent

    @property
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        return self._metadata_info

    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        history_last_access_iter = history_metadata["last_access_iter"]
        history_last_access_iter[:] = current_iter

    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
        threshold_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coalesced_history_metadata: Dict[str, torch.Tensor] = {}
        history_last_access_iter = history_metadata["last_access_iter"]
        if additional_ids is not None:
            history_last_access_iter = torch.cat(
                [
                    history_last_access_iter,
                    torch.full_like(additional_ids, current_iter),
                ]
            )
        coalesced_history_metadata["last_access_iter"] = torch.zeros_like(
            unique_ids_counts
        ).scatter_reduce_(
            0,
            unique_inverse_mapping,
            history_last_access_iter,
            reduce="amax",
            include_self=False,
        )
        if threshold_mask is not None:
            coalesced_history_metadata["last_access_iter"] = coalesced_history_metadata[
                "last_access_iter"
            ][threshold_mask]
        return coalesced_history_metadata

    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mch_counts = mch_metadata["counts"]
        mch_last_access_iter = mch_metadata["last_access_iter"]

        # sort coalesced history metadata
        coalesced_history_metadata["last_access_iter"].copy_(
            coalesced_history_metadata["last_access_iter"][
                coalesced_history_argsort_mapping
            ]
        )
        coalesced_history_sorted_uniq_ids_last_access_iter = coalesced_history_metadata[
            "last_access_iter"
        ]

        # update metadata for matching ids
        mch_counts[
            coalesced_history_mch_matching_indices
        ] += coalesced_history_sorted_unique_ids_counts[
            coalesced_history_mch_matching_elements_mask
        ]
        mch_last_access_iter[coalesced_history_mch_matching_indices] = (
            coalesced_history_sorted_uniq_ids_last_access_iter[
                coalesced_history_mch_matching_elements_mask
            ]
        )

        # incoming non-matching ids
        new_sorted_uniq_ids_counts = coalesced_history_sorted_unique_ids_counts[
            ~coalesced_history_mch_matching_elements_mask
        ]
        new_sorted_uniq_ids_last_access = (
            coalesced_history_sorted_uniq_ids_last_access_iter[
                ~coalesced_history_mch_matching_elements_mask
            ]
        )

        # TODO: find cleaner way to avoid last element of zch
        mch_counts[mch_size - 1] = torch.iinfo(torch.int64).max
        mch_last_access_iter[mch_size - 1] = current_iter

        merged_counts = torch.cat(
            [
                mch_counts,
                new_sorted_uniq_ids_counts,
            ]
        )
        merged_access_iter = torch.cat(
            [
                mch_last_access_iter,
                new_sorted_uniq_ids_last_access,
            ]
        )
        merged_weighted_distance = torch.pow(
            current_iter - merged_access_iter + 1,
            self._decay_exponent,
        )
        # merged eviction scores are the eviction scores calculated for the
        #   tensor torch.cat[_mch_sorted_raw_ids, frequency_sorted_uniq_ids[~matching_eles]]
        # lower scores are evicted first.
        merged_eviction_scores = torch.div(merged_counts, merged_weighted_distance)

        # calculate evicted and replacement indices
        (
            evicted_indices,
            selected_new_indices,
        ) = self._compute_selected_eviction_and_replacement_indices(
            mch_size,
            merged_eviction_scores,
        )

        # update metadata for evicted ids
        mch_counts[evicted_indices] = new_sorted_uniq_ids_counts[selected_new_indices]

        mch_last_access_iter[evicted_indices] = new_sorted_uniq_ids_last_access[
            selected_new_indices
        ]

        return evicted_indices, selected_new_indices


@torch.fx.wrap
def _mch_remap(
    features: Dict[str, JaggedTensor],
    mch_sorted_raw_ids: torch.Tensor,
    mch_remapped_ids_mapping: torch.Tensor,
    zch_index: int,
) -> Dict[str, JaggedTensor]:
    """Remap feature ids to zch ids, TODO: create a custom kernel"""
    remapped_features: Dict[str, JaggedTensor] = {}
    for name, feature in features.items():
        values = feature.values()
        remapped_ids = torch.empty_like(values)

        # compute overlap between incoming IDs and remapping table
        searched_indices = torch.searchsorted(mch_sorted_raw_ids[:-1], values)
        retrieved_indices = mch_sorted_raw_ids[searched_indices]
        # identify matching inputs IDs
        matching_indices = retrieved_indices == values
        # update output with remapped matching IDs
        remapped_ids[matching_indices] = mch_remapped_ids_mapping[
            searched_indices[matching_indices]
        ]
        # default embedding for non-matching ids
        remapped_ids[~matching_indices] = zch_index

        remapped_features[name] = JaggedTensor(
            values=remapped_ids,
            lengths=feature.lengths(),
            offsets=feature.offsets(),
            weights=feature.weights_or_none(),
        )
    return remapped_features


class MCHManagedCollisionModule(ManagedCollisionModule):
    """
    ZCH managed collision module

    Args:
        zch_size (int): range of output ids, within [output_size_offset, output_size_offset + zch_size - 1)
        device (torch.device): device on which this module will be executed
        eviction_policy (eviction policy): eviction policy to be used
        eviction_interval (int): interval of eviction policy is triggered
        input_hash_size (int): input feature id range, will be passed to input_hash_func as second arg
        input_hash_func (Optional[Callable]): function used to generate hashes for input features.  This function is typically used to drive uniform distribution over range same or greater than input data
        mch_size (Optional[int]): DEPRECIATED - size of residual output (ie. legacy MCH), experimental feature.  Ids are internally shifted by output_size_offset + zch_output_range
        mch_hash_func (Optional[Callable]): DEPRECIATED - function used to generate hashes for residual feature. will hash down to mch_size.
        output_global_offset (int): offset of the output id for output range, typically only used in sharding applications.
    """

    def __init__(
        self,
        zch_size: int,
        device: torch.device,
        eviction_policy: MCHEvictionPolicy,
        eviction_interval: int,
        input_hash_size: int = (2**63) - 1,
        input_hash_func: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        mch_size: Optional[int] = None,
        mch_hash_func: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        name: Optional[str] = None,
        output_global_offset: int = 0,  # typically not provided by user
        output_segments: Optional[List[int]] = None,  # typically not provided by user
        buckets: int = 1,
    ) -> None:
        if output_segments is None:
            output_segments = [output_global_offset, output_global_offset + zch_size]
        super().__init__(
            device=device,
            output_segments=output_segments,
        )
        if mch_size is not None or mch_hash_func is not None:
            logger.warning(
                "co-locating a hash table for missing ids is depreciated (ie. mch_size, mch_hash_func), values will be ignored"
            )
        self._init_output_segments_tensor: torch.Tensor = self._output_segments_tensor
        self._name = name
        self._input_history_buffer_size: int = -1
        self._input_hash_size = input_hash_size
        self._zch_size: int = zch_size
        assert self._zch_size > 0, "zch_size must be > 0"
        self._output_global_offset: int = output_global_offset
        self._input_hash_func = input_hash_func

        self._eviction_interval = eviction_interval
        assert self._eviction_interval > 0, "eviction_interval must be > 1"
        self._eviction_policy = eviction_policy

        self._current_iter: int = -1
        self._buckets = buckets
        self._init_buffers()

        ## ------ history info ------
        self._mch_metadata: Dict[str, torch.Tensor] = {}
        self._history_metadata: Dict[str, torch.Tensor] = {}
        self._init_metadata_buffers()
        self._current_history_buffer_offset: int = 0

        self._evicted: bool = False
        self._last_eviction_iter: int = -1

    def _init_buffers(self) -> None:
        self.register_buffer(
            "_mch_sorted_raw_ids",
            torch.full(
                (self._zch_size,),
                torch.iinfo(torch.int64).max,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_mch_slots",
            torch.tensor(
                [(self._zch_size - 1)],
                dtype=torch.int64,
                device=self.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_delimiter",
            torch.tensor(
                [torch.iinfo(torch.int64).max], dtype=torch.int64, device=self.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "_mch_remapped_ids_mapping",
            torch.arange(
                start=self._output_global_offset,
                end=self._output_global_offset + self._zch_size,
                dtype=torch.int64,
                device=self.device,
            ),
        )

        self._evicted_emb_indices: torch.Tensor = torch.empty((1,), device=self.device)

    def _init_metadata_buffers(self) -> None:
        eviction_metadata_info = self._eviction_policy.metadata_info
        for metadata in eviction_metadata_info:
            metadata_name, is_mch_metadata, is_history_metadata = metadata
            # mch_metadata
            if is_mch_metadata:
                buffer_name = "_mch_" + metadata_name
                self.register_buffer(
                    buffer_name,
                    torch.zeros(
                        (self._zch_size,),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )
                self._mch_metadata[metadata_name] = getattr(self, buffer_name)

    def _init_history_buffers(self, features: Dict[str, JaggedTensor]) -> None:
        input_batch_value_size_cumsum = 0
        for _, feature in features.items():
            input_batch_value_size_cumsum += feature.values().numel()
        self._input_history_buffer_size = int(
            input_batch_value_size_cumsum * self._eviction_interval * 1.25
        )
        # pyre-fixme[16]: `MCHManagedCollisionModule` has no attribute
        #  `_history_accumulator`.
        self._history_accumulator: torch.Tensor = torch.empty(
            self._input_history_buffer_size,
            dtype=torch.int64,
            device=self.device,
        )
        eviction_metadata_info = self._eviction_policy.metadata_info
        for metadata in eviction_metadata_info:
            metadata_name, is_mch_metadata, is_history_metadata = metadata
            # history_metadata
            if is_history_metadata:
                buffer_name = "_history_" + metadata_name
                self.register_buffer(
                    buffer_name,
                    torch.zeros(
                        self._input_history_buffer_size,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    persistent=False,
                )
                self._history_metadata[metadata_name] = getattr(self, buffer_name)

    def preprocess(self, features: Dict[str, JaggedTensor]) -> Dict[str, JaggedTensor]:
        if self._input_hash_func is None:
            return features
        preprocessed_features: Dict[str, JaggedTensor] = {}
        for name, feature in features.items():
            preprocessed_features[name] = JaggedTensor(
                # pyre-ignore [29]
                values=self._input_hash_func(feature.values(), self._input_hash_size),
                lengths=feature.lengths(),
                offsets=feature.offsets(),
                weights=feature.weights_or_none(),
            )
        return preprocessed_features

    def reset_inference_mode(self) -> None:
        self._evicted = False
        self._last_eviction_iter = -1

    @torch.no_grad()
    def _match_indices(
        self, sorted_sequence: torch.Tensor, search_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        searched_indices = torch.searchsorted(sorted_sequence[:-1], search_values)
        retrieved_ids = sorted_sequence[searched_indices]
        matching_eles = retrieved_ids == search_values
        matched_indices = searched_indices[matching_eles]
        return (matching_eles, matched_indices)

    @torch.no_grad()
    def _sort_mch_buffers(self) -> None:
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Module,
        #  Tensor]`.
        argsorted_sorted_raw_ids = torch.argsort(self._mch_sorted_raw_ids, stable=True)
        # pyre-fixme[29]: `Union[(self: TensorBase, src: Tensor, non_blocking: bool
        #  = ...) -> Tensor, Module, Tensor]` is not a function.
        self._mch_sorted_raw_ids.copy_(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            self._mch_sorted_raw_ids[argsorted_sorted_raw_ids]
        )
        # pyre-fixme[29]: `Union[(self: TensorBase, src: Tensor, non_blocking: bool
        #  = ...) -> Tensor, Module, Tensor]` is not a function.
        self._mch_remapped_ids_mapping.copy_(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            self._mch_remapped_ids_mapping[argsorted_sorted_raw_ids]
        )
        for mch_metadata_buffer in self._mch_metadata.values():
            mch_metadata_buffer.copy_(mch_metadata_buffer[argsorted_sorted_raw_ids])

    @torch.no_grad()
    def _update_and_evict(
        self,
        uniq_ids: torch.Tensor,
        uniq_ids_counts: torch.Tensor,
        uniq_ids_metadata: Dict[str, torch.Tensor],
    ) -> None:
        argsorted_uniq_ids_counts = torch.argsort(
            uniq_ids_counts, descending=True, stable=True
        )
        frequency_sorted_uniq_ids = uniq_ids[argsorted_uniq_ids_counts]
        frequency_sorted_uniq_ids_counts = uniq_ids_counts[argsorted_uniq_ids_counts]

        matching_eles, matched_indices = self._match_indices(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            self._mch_sorted_raw_ids,
            frequency_sorted_uniq_ids,
        )

        new_frequency_sorted_uniq_ids = frequency_sorted_uniq_ids[~matching_eles]

        # evicted_indices are indices in the mch map to be evicted, and
        #   selected_new_indices are the indices of the ids in the coalesced
        #   history that are to be added to the mch.
        (
            evicted_indices,
            selected_new_indices,
        ) = self._eviction_policy.update_metadata_and_generate_eviction_scores(
            self._current_iter,
            self._zch_size,
            argsorted_uniq_ids_counts,
            frequency_sorted_uniq_ids_counts,
            matching_eles,
            matched_indices,
            self._mch_metadata,
            uniq_ids_metadata,
        )
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        self._mch_sorted_raw_ids[evicted_indices] = new_frequency_sorted_uniq_ids[
            selected_new_indices
        ]

        # NOTE evicted ids for emb reset
        # if evicted flag is already set, then existing evicted ids havent been
        # consumed by evict(). append new evicted ids to the list
        if self._evicted:
            self._evicted_emb_indices = torch.unique(
                torch.cat(
                    [
                        self._evicted_emb_indices,
                        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[No...
                        self._mch_remapped_ids_mapping[evicted_indices],
                    ]
                )
            )
        else:
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            self._evicted_emb_indices = self._mch_remapped_ids_mapping[evicted_indices]
        self._evicted = True

        # re-sort for next search
        self._sort_mch_buffers()

    @torch.no_grad()
    def _coalesce_history(self) -> None:
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        current_history_accumulator = self._history_accumulator[
            : self._current_history_buffer_offset
        ]
        uniq_ids, uniq_inverse_mapping, uniq_ids_counts = torch.unique(
            current_history_accumulator,
            return_inverse=True,
            return_counts=True,
        )
        if self._eviction_policy._threshold_filtering_func is not None:
            threshold_mask, threshold = self._eviction_policy._threshold_filtering_func(
                uniq_ids_counts
            )
        else:
            threshold_mask = None

        coalesced_eviction_history_metadata = (
            self._eviction_policy.coalesce_history_metadata(
                self._current_iter,
                {
                    metadata_name: metadata_buffer[
                        : self._current_history_buffer_offset
                    ]
                    for metadata_name, metadata_buffer in self._history_metadata.items()
                },
                uniq_ids_counts,
                uniq_inverse_mapping,
                threshold_mask=threshold_mask,
            )
        )
        if threshold_mask is not None:
            uniq_ids = uniq_ids[threshold_mask]
            uniq_ids_counts = uniq_ids_counts[threshold_mask]
        self._update_and_evict(
            uniq_ids, uniq_ids_counts, coalesced_eviction_history_metadata
        )
        # reset buffer offset
        self._current_history_buffer_offset = 0

    def profile(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        if not self.training:
            return features

        if self._current_iter == -1:
            self._current_iter = int(self._current_iter_tensor.item())
            self._last_eviction_iter = self._current_iter
        self._current_iter += 1
        self._current_iter_tensor.data += 1

        # init history buffers if needed
        if self._input_history_buffer_size == -1:
            self._init_history_buffers(features)

        for _, feature in features.items():
            values = feature.values()
            free_elements = (
                self._input_history_buffer_size - self._current_history_buffer_offset
            )
            values = values[:free_elements]
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            self._history_accumulator[
                self._current_history_buffer_offset : self._current_history_buffer_offset
                + values.shape[0]
            ] = values
            self._eviction_policy.record_history_metadata(
                self._current_iter,
                values,
                {
                    metadata_name: metadata_buffer[
                        self._current_history_buffer_offset : self._current_history_buffer_offset
                        + values.shape[0]
                    ]
                    for metadata_name, metadata_buffer in self._history_metadata.items()
                },
            )
            self._current_history_buffer_offset += values.shape[0]

        # coalesce history / evict
        if self._current_iter - self._last_eviction_iter == self._eviction_interval:
            self._coalesce_history()
            self._last_eviction_iter = self._current_iter

        return features

    def remap(self, features: Dict[str, JaggedTensor]) -> Dict[str, JaggedTensor]:
        return _mch_remap(
            features,
            self._mch_sorted_raw_ids,
            self._mch_remapped_ids_mapping,
            self._output_global_offset + self._zch_size - 1,
        )

    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
        feature (JaggedTensor]): feature representation
        Returns:
            Dict[str, JaggedTensor]: modified JT
        """

        features = self.preprocess(features)
        features = self.profile(features)
        return self.remap(features)

    def output_size(self) -> int:
        return self._zch_size

    def buckets(self) -> int:
        return self._buckets

    def input_size(self) -> int:
        return self._input_hash_size

    def open_slots(self) -> torch.Tensor:
        # pyre-fixme[29]: `Union[(self: TensorBase, other: Any) -> Tensor, Module,
        #  Tensor]` is not a function.
        return self._mch_slots - torch.searchsorted(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            self._mch_sorted_raw_ids,
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            self._delimiter,
        )

    @torch.no_grad()
    def evict(self) -> Optional[torch.Tensor]:
        if self._evicted:
            self._evicted = False
            return self._evicted_emb_indices
        else:
            return None

    def validate_state(self) -> None:
        start = self._output_global_offset
        end = start + self._zch_size
        assert (
            start in self._output_segments_tensor
            and end in self._output_segments_tensor
        ), f"shard within range [{start}, {end}] cannot be built out of segements {self._output_segments_tensor}"

        # update output segments and resort
        self._output_segments_tensor = self._init_output_segments_tensor
        self._sort_mch_buffers()

    def rebuild_with_output_id_range(
        self,
        output_id_range: Tuple[int, int],
        output_segments: List[int],
        device: Optional[torch.device] = None,
    ) -> "MCHManagedCollisionModule":

        new_zch_size = output_id_range[1] - output_id_range[0]

        return type(self)(
            name=self._name,
            zch_size=new_zch_size,
            device=device or self.device,
            eviction_policy=self._eviction_policy,
            eviction_interval=self._eviction_interval,
            input_hash_size=self._input_hash_size,
            input_hash_func=self._input_hash_func,
            output_global_offset=output_id_range[0],
            output_segments=output_segments,
            buckets=len(output_segments) - 1,
        )
