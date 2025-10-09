#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fbgemm_gpu  # @manual = "//deeplearning/fbgemm/fbgemm_gpu:fbgemm_gpu"

import torch

from torchrec.modules.hash_mc_evictions import (
    get_kernel_from_policy,
    HashZchEvictionConfig,
    HashZchEvictionModule,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_metrics import ScalarLogger
from torchrec.modules.mc_modules import ManagedCollisionModule
from torchrec.sparse.jagged_tensor import JaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


@torch.fx.wrap
def _tensor_may_to_device(
    src: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.device]:
    src_device: torch.device = src.device
    if device is None:
        return (src, src_device)

    if device.type != "meta" and src_device != device:
        return (src.to(device), src_device)
    return (src, src_device)


class TrainInputMapper(torch.nn.Module):
    """
    Module used to generate sizes and offsets information corresponding to
    the train ranks for inference inputs. This is due to we currently merge
    all identity tensors that are row-wise sharded across training ranks at
    inference time. So we need to map the inputs to the chunk of identities
    that the input would go at training time to generate appropriate indices.

    Args:
        input_hash_size: the max size of input IDs
        total_num_buckets: the total number of buckets across all ranks at training time
        size_per_rank: the size of the identity tensor/embedding size per rank
        train_rank_offsets: the offset of the embedding table indices per rank
        inference_dispatch_div_train_world_size: the flag to control whether to divide input by
            world_size https://fburl.com/code/c9x98073
        name: the name of the embedding table

    Example::
        mapper = TrainInputMapper(...)
        mapper(values, output_offset)
    """

    def __init__(
        self,
        input_hash_size: int,
        total_num_buckets: int,
        size_per_rank: torch.Tensor,
        train_rank_offsets: torch.Tensor,
        inference_dispatch_div_train_world_size: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._input_hash_size = input_hash_size
        assert total_num_buckets > 0, f"{total_num_buckets=} must be positive"
        self._buckets = total_num_buckets
        self._inference_dispatch_div_train_world_size = (
            inference_dispatch_div_train_world_size
        )
        self._name = name
        self.register_buffer(
            "_zch_size_per_training_rank", size_per_rank, persistent=False
        )
        self.register_buffer(
            "_train_rank_offsets", train_rank_offsets, persistent=False
        )
        logger.info(
            f"TrainInputMapper: {self._name=}, {self._input_hash_size=}, {self._zch_size_per_training_rank=}, "
            f"{self._train_rank_offsets=}, {self._inference_dispatch_div_train_world_size=}"
        )

    # TODO: make a kernel
    def _get_values_sizes_offsets(
        self, x: torch.Tensor, output_offset: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        zch_size_per_training_rank, _ = _tensor_may_to_device(
            self._zch_size_per_training_rank, x.device
        )
        train_rank_offsets, _ = _tensor_may_to_device(
            self._train_rank_offsets, x.device
        )

        # NOTE: This assumption has to be the same as TorchRec input_dist logic
        # https://fburl.com/code/c9x98073. Do not use torch.where() for performance.
        if self._input_hash_size == 0:
            train_ranks = x % self._buckets
            if self._inference_dispatch_div_train_world_size:
                x = x // self._buckets
        else:
            blk_size = (self._input_hash_size // self._buckets) + (
                0 if self._input_hash_size % self._buckets == 0 else 1
            )
            train_ranks = x // blk_size
            if self._inference_dispatch_div_train_world_size:
                x = x % blk_size

        local_sizes = zch_size_per_training_rank.index_select(dim=0, index=train_ranks)
        offsets = train_rank_offsets.index_select(dim=0, index=train_ranks)
        if output_offset is not None:
            offsets -= output_offset

        return (x, local_sizes, offsets)

    def forward(
        self,
        values: torch.Tensor,
        output_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            values: ID values to compute bucket assignment and offset.
            output_offset: global offset of the start bucket per rank, used to compute bucket offset within the rank.

        Returns:
            A tuple of three tensors:
                - values: transformed ID values, different from input value only if inference_dispatch_div_train_world_size is True.
                - local_sizes: bucket sizes of the input values.
                - offsets: in-rank bucket offsets of the input values.
        """

        values, local_sizes, offsets = self._get_values_sizes_offsets(
            values, output_offset
        )
        return (values, local_sizes, offsets)


@torch.fx.wrap
def _get_device(hash_zch_identities: torch.Tensor) -> torch.device:
    return hash_zch_identities.device


class HashZchManagedCollisionModule(ManagedCollisionModule):
    """
    Module to manage multi-probe ZCH (MPZCH), including lookup (remapping), eviction, metrics collection, and required auxiliary tensors.

    Args:
        zch_size: local size of the embedding table
        device: the compute device
        total_num_buckets: logical shard within each rank for resharding purpose, note that
            1) zch_size must be a multiple of total_num_buckets, and 2) total_num_buckets must be a multiple of world size
        max_probe: the number of times MPZCH kernel attempts to run linear search for lookup or insertion
        input_hash_size: the max size of input IDs (default to 0)
        output_segments: the index range of each bucket, which is computed before sharding and typically not provided by user
        is_inference: the flag to indicate if the module is used in inference as opposed to train or eval
        name: the name of the embedding table
        tb_logging_frequency: the frequency of emitting metrics to TensorBoard, measured by the number of batches
        eviction_policy_name: the specific policy to be used for eviction operations
        eviction_config: the config associated with the selected eviction policy
        inference_dispatch_div_train_world_size: the flag to control whether to divide input by
            world_size https://fburl.com/code/c9x98073
        start_bucket: start bucket of the current rank, typically not provided by user
        end_bucket: end bucket of the current rank, typically not provided by user
        opt_in_prob: the probability of an ID to be opted in from a statistical aspect
        percent_reserved_slots: percentage of slots to be reserved when opt-in is enabled, the value must be in [0, 100)

    Example::
        module = HashZchManagedCollisionModule(...)
        module(features)
    """

    _evicted_indices: List[torch.Tensor]

    IDENTITY_BUFFER: str = "_hash_zch_identities"
    METADATA_BUFFER: str = "_hash_zch_metadata"

    table_name_on_device_remapped_ids_dict: Dict[
        str, torch.Tensor
    ]  # used to store the on-device remapped ids
    table_name_on_device_input_ids_dict: Dict[
        str, torch.Tensor
    ]  # used to store the on-device input ids

    def __init__(
        self,
        zch_size: int,
        device: torch.device,
        total_num_buckets: int,
        max_probe: int = 128,
        input_hash_size: int = 0,
        output_segments: Optional[List[int]] = None,
        is_inference: bool = False,
        name: Optional[str] = None,
        tb_logging_frequency: int = 0,
        eviction_policy_name: Optional[HashZchEvictionPolicyName] = None,
        eviction_config: Optional[HashZchEvictionConfig] = None,
        inference_dispatch_div_train_world_size: bool = False,
        start_bucket: int = 0,
        end_bucket: Optional[int] = None,
        opt_in_prob: int = -1,
        percent_reserved_slots: float = 0,
        disable_fallback: bool = False,
    ) -> None:
        if output_segments is None:
            assert (
                zch_size % total_num_buckets == 0
            ), f"please pass output segments if not uniform buckets {zch_size=}, {total_num_buckets=}"
            output_segments = [
                (zch_size // total_num_buckets) * bucket
                for bucket in range(total_num_buckets + 1)
            ]

        super().__init__(
            device=device,
            output_segments=output_segments,
            skip_state_validation=True,  # avoid peristent buffers for TGIF Puslishing
        )

        self._zch_size: int = zch_size
        self._output_segments: List[int] = output_segments
        self._start_bucket: int = start_bucket
        self._end_bucket: int = (
            end_bucket if end_bucket is not None else total_num_buckets
        )
        self._output_global_offset_tensor: Optional[torch.Tensor] = None
        if output_segments[start_bucket] > 0:
            self._output_global_offset_tensor = torch.tensor(
                [output_segments[start_bucket]],
                dtype=torch.int64,
                device=device if device.type != "meta" else torch.device("cpu"),
            )

        self._device: torch.device = device
        self._input_hash_size: int = input_hash_size
        self._is_inference: bool = is_inference
        self._name: Optional[str] = name
        self._tb_logging_frequency: int = tb_logging_frequency
        self._scalar_logger: Optional[ScalarLogger] = None
        self._eviction_policy_name: Optional[HashZchEvictionPolicyName] = (
            eviction_policy_name
        )
        self._eviction_config: Optional[HashZchEvictionConfig] = eviction_config
        self._eviction_module: Optional[HashZchEvictionModule] = (
            HashZchEvictionModule(
                policy_name=self._eviction_policy_name,
                device=self._device,
                config=self._eviction_config,
            )
            if self._eviction_policy_name is not None and self.training
            else None
        )
        self._opt_in_prob: int = opt_in_prob
        assert (
            percent_reserved_slots >= 0 and percent_reserved_slots < 100
        ), "percent_reserved_slots must be in [0, 100)"
        self._percent_reserved_slots: float = percent_reserved_slots
        if self._opt_in_prob > 0:
            assert (
                self._percent_reserved_slots > 0
            ), "percent_reserved_slots must be positive when opt_in_prob is positive"
            assert (
                self._eviction_policy_name is None
                or self._eviction_policy_name != HashZchEvictionPolicyName.LRU_EVICTION
            ), "LRU eviction is not compatible with opt-in at this time"
        self._disable_fallback: bool = disable_fallback

        if torch.jit.is_scripting() or self._is_inference or self._name is None:
            self._tb_logging_frequency = 0

        if self._tb_logging_frequency > 0 and self._device.type != "meta":
            assert self._name is not None
            self._scalar_logger = ScalarLogger(
                name=self._name,
                zch_size=self._zch_size,
                frequency=self._tb_logging_frequency,
                start_bucket=self._start_bucket,
                disable_fallback=self._disable_fallback,
            )
        else:
            logger.info(
                f"ScalarLogger is disabled because {self._tb_logging_frequency=} and {self._device.type=}"
            )

        identities, metadata = torch.ops.fbgemm.create_zch_buffer(
            size=self._zch_size,
            support_evict=self._eviction_module is not None,
            device=self._device,
            long_type=True,  # deprecated, always True
        )

        self._hash_zch_identities = torch.nn.Parameter(identities, requires_grad=False)
        self.register_buffer(HashZchManagedCollisionModule.METADATA_BUFFER, metadata)

        self._max_probe = max_probe
        self._buckets = total_num_buckets
        self.register_buffer("_hash_zch_bucket", torch.tensor([total_num_buckets]))
        # Do not need to store in buffer since this is created and consumed
        # at each step https://fburl.com/code/axzimmbx
        self._evicted_indices = []

        # do not pass device, so its initialized on default physical device ('meta' will result in silent failure)
        size_per_rank = torch.diff(
            torch.tensor(self._output_segments, dtype=torch.int64)
        )

        self.input_mapper: torch.nn.Module = TrainInputMapper(
            input_hash_size=self._input_hash_size,
            total_num_buckets=total_num_buckets,
            size_per_rank=size_per_rank,
            train_rank_offsets=torch.tensor(
                torch.ops.fbgemm.asynchronous_exclusive_cumsum(size_per_rank)
            ),
            # be consistent with https://fburl.com/code/p4mj4mc1
            inference_dispatch_div_train_world_size=inference_dispatch_div_train_world_size,
            name=self._name,
        )

        if self._is_inference is True:
            self.reset_inference_mode()

        self._eviction_policy_name_copy: Optional[HashZchEvictionPolicyName] = (
            self._eviction_policy_name
        )

        # create two dictionaries to store the input values and remapped ids on the current rank
        # these values are used for calculating zch metrics like hit rate and collision rate
        ## on-device remapped ids
        self.table_name_on_device_remapped_ids_dict: Dict[str, torch.Tensor] = (
            {}
        )  # {table_name: on_device_remapped_ids}
        ## on-device input ids
        self.table_name_on_device_input_ids_dict: Dict[str, torch.Tensor] = (
            {}
        )  # {table_name: input JT values that maps to the current rank}

        logger.info(
            f"HashZchManagedCollisionModule: {self._name=}, {self.device=}, "
            f"{self._zch_size=}, {self._input_hash_size=}, {self._max_probe=}, "
            f"{self._is_inference=}, {self._tb_logging_frequency=}, "
            f"{self._eviction_policy_name=}, {self._eviction_config=}, "
            f"{self._buckets=}, {self._start_bucket=}, {self._end_bucket=}, "
            f"{self._output_global_offset_tensor=}, {self._output_segments=}, "
            f"{inference_dispatch_div_train_world_size=}, "
            f"{self._opt_in_prob=}, {self._percent_reserved_slots=}, {self._disable_fallback=}"
        )

    @property
    def device(self) -> torch.device:
        return _get_device(self._hash_zch_identities)

    def buckets(self) -> int:
        return self._buckets

    # TODO: This is hacky as we are using parameters to go through publishing.
    # Can remove once working out buffer solution.
    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from super().named_buffers(prefix, recurse, remove_duplicate)
        key: str = HashZchManagedCollisionModule.IDENTITY_BUFFER
        if prefix:
            key = f"{prefix}.{key}"
        yield (key, self._hash_zch_identities.data)

    def validate_state(self) -> None:
        raise NotImplementedError()

    def _set_eval_mode_impl(self) -> None:
        # self.eval() sets self.training = false
        self.eval()
        self._is_inference = True
        self._evicted_indices = []
        # disable eviction
        self._eviction_policy_name = None
        self._eviction_module = None

    def reset_inference_mode(
        self,
    ) -> None:
        logger.info("HashZchManagedCollisionModule resetting inference mode")
        self._set_eval_mode_impl()
        self._hash_zch_metadata = None

        def _load_state_dict_pre_hook(
            module: "HashZchManagedCollisionModule",
            state_dict: Dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> None:
            logger.info("HashZchManagedCollisionModule loading state dict")
            # We store the full identity in checkpoint and predictor, cut it at inference loading
            if not self._is_inference:
                return
            if "_hash_zch_metadata" in state_dict:
                del state_dict["_hash_zch_metadata"]

        self._register_load_state_dict_pre_hook(
            _load_state_dict_pre_hook, with_module=True
        )

    def reset_intrainer_bulk_eval_mode(self) -> None:
        logger.info(
            "HashZchManagedCollisionModule resetting to intrainer bulk eval mode"
        )
        self._set_eval_mode_impl()

    def reset_training_mode(
        self,
    ) -> None:
        logger.info("HashZchManagedCollisionModule resetting to training mode")
        self._is_inference = False
        self.train()
        self._eviction_policy_name = self._eviction_policy_name_copy
        self._eviction_module = (
            HashZchEvictionModule(
                policy_name=self._eviction_policy_name,
                device=self._device,
                config=self._eviction_config,
            )
            if self._eviction_policy_name is not None
            else None
        )

    def preprocess(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        return features

    def evict(self) -> Optional[torch.Tensor]:
        if len(self._evicted_indices) == 0:
            return None
        out = torch.unique(torch.cat(self._evicted_indices))
        self._evicted_indices = []
        return (
            out + self._output_global_offset_tensor
            if self._output_global_offset_tensor
            else out
        )

    def profile(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        return features

    def get_reserved_slots_per_bucket(self) -> int:
        if self._opt_in_prob == -1:
            return -1

        return math.floor(
            self._zch_size
            * self._percent_reserved_slots
            / 100
            / (self._end_bucket - self._start_bucket)
        )

    def remap(self, features: Dict[str, JaggedTensor]) -> Dict[str, JaggedTensor]:
        readonly: bool = False
        if self._output_global_offset_tensor is not None:
            self._output_global_offset_tensor, _ = _tensor_may_to_device(
                self._output_global_offset_tensor, self.device
            )

        metadata: Optional[torch.Tensor] = None
        if self.training:
            metadata = self._hash_zch_metadata
        else:
            readonly = True

        # _evicted_indices will be reset in evict(): https://fburl.com/code/r3fxcs1y
        assert len(self._evicted_indices) == 0

        # `torch.no_grad()` Annotatin prevents torchscripting `JaggedTensor` for some reason...
        with torch.no_grad():
            remapped_features: Dict[str, JaggedTensor] = {}
            identities_0 = (
                self._hash_zch_identities.data.clone()
                # if self._tb_logging_frequency > 0
                # else None
            )

            for name, feature in features.items():
                values = feature.values()
                input_metadata, eviction_threshold = (
                    self._eviction_module(feature)
                    if self._eviction_module is not None
                    else (None, -1)
                )

                opt_in_rands = (
                    (torch.rand_like(values, dtype=torch.float) * 100).to(torch.int32)
                    if self._opt_in_prob != -1 and self.training
                    else None
                )

                values, orig_device = _tensor_may_to_device(values, self.device)
                values, local_sizes, offsets = self.input_mapper(
                    values=values,
                    output_offset=self._output_global_offset_tensor,
                )

                # record the input values
                self.table_name_on_device_input_ids_dict[name] = values.clone()

                num_reserved_slots = self.get_reserved_slots_per_bucket()
                remapped_ids, evictions = torch.ops.fbgemm.zero_collision_hash(
                    input=values,
                    identities=self._hash_zch_identities,
                    max_probe=self._max_probe,
                    circular_probe=True,
                    exp_hours=-1,  # deprecated, always -1
                    readonly=readonly,
                    local_sizes=local_sizes,
                    offsets=offsets,
                    metadata=metadata,
                    # Use self._is_inference to turn on writing to pinned
                    # CPU memory directly. But may not have perf benefit.
                    output_on_uvm=False,  # self._is_inference,
                    disable_fallback=self._disable_fallback,
                    _modulo_identity_DPRECATED=False,  # deprecated, always False
                    input_metadata=input_metadata,
                    eviction_threshold=eviction_threshold,
                    eviction_policy=get_kernel_from_policy(self._eviction_policy_name),
                    opt_in_prob=self._opt_in_prob,
                    num_reserved_slots=num_reserved_slots,
                    opt_in_rands=opt_in_rands,
                )

                # record the on-device remapped ids
                self.table_name_on_device_remapped_ids_dict[name] = remapped_ids.clone()
                lengths: torch.Tensor = feature.lengths()
                if self._disable_fallback:
                    # Only works on GPU when read only is true.
                    hit_indices = remapped_ids != -1
                    remapped_ids = remapped_ids[hit_indices]
                    lengths = torch.masked_fill(lengths, ~hit_indices, 0)
                if self._scalar_logger is not None:
                    assert identities_0 is not None
                    self._scalar_logger.update(
                        identities_0=identities_0,
                        identities_1=self._hash_zch_identities,
                        values=values,
                        remapped_ids=remapped_ids,
                        evicted_emb_indices=evictions,
                        metadata=metadata,
                        num_reserved_slots=num_reserved_slots,
                        eviction_config=self._eviction_config,
                    )

                output_global_offset_tensor = self._output_global_offset_tensor
                if output_global_offset_tensor is not None:
                    remapped_ids = remapped_ids + output_global_offset_tensor

                _append_eviction_indice(self._evicted_indices, evictions)
                remapped_ids, _ = _tensor_may_to_device(remapped_ids, orig_device)

                remapped_features[name] = JaggedTensor(
                    values=remapped_ids,
                    lengths=lengths,
                    offsets=feature.offsets(),
                    weights=feature.weights_or_none(),
                )

            if self._scalar_logger is not None:
                self._scalar_logger(
                    run_type="train" if self.training else "eval",
                    identities=self._hash_zch_identities.data,
                )

            return remapped_features

    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        return self.remap(features)

    def output_size(self) -> int:
        return self._zch_size

    def input_size(self) -> int:
        return self._input_hash_size

    def open_slots(self) -> torch.Tensor:
        return torch.tensor([0])

    def rebuild_with_output_id_range(
        self,
        output_id_range: Tuple[int, int],
        output_segments: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> "HashZchManagedCollisionModule":
        # rebuild should use existing output_segments instead of the input one and should not
        # recalculate since the output segments are calculated based on the original embedding
        # table size, total bucket number, which might not be available for the rebuild caller
        try:
            start_idx = self._output_segments.index(output_id_range[0])
            end_idx = self._output_segments.index(output_id_range[1])
        except ValueError:
            raise RuntimeError(
                f"Attempting to shard HashZchManagedCollisionModule, but rank {device} does not align with bucket boundaries;"
                + f" please check kwarg total_num_buckets={self._buckets} is a multiple of world size."
            )
        new_zch_size = output_id_range[1] - output_id_range[0]

        return self.__class__(
            zch_size=new_zch_size,
            device=device or self.device,
            max_probe=self._max_probe,
            total_num_buckets=self._buckets,
            input_hash_size=self._input_hash_size,
            is_inference=self._is_inference,
            start_bucket=start_idx,
            end_bucket=end_idx,
            output_segments=self._output_segments,
            name=self._name,
            tb_logging_frequency=self._tb_logging_frequency,
            eviction_policy_name=self._eviction_policy_name,
            eviction_config=self._eviction_config,
            opt_in_prob=self._opt_in_prob,
            percent_reserved_slots=self._percent_reserved_slots,
            disable_fallback=self._disable_fallback,
        )


@torch.fx.wrap
def _append_eviction_indice(
    evicted_indices: List[torch.Tensor],
    evictions: Optional[torch.Tensor],
) -> None:
    if evictions is not None and evictions.numel() > 0:
        evicted_indices.append(evictions)
