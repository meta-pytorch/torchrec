#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import pdb  # noqa
import sys
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torch.autograd.profiler import record_function
from torchrec import optim as trec_optim
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    KeyedJaggedTensor,
)
from torchrec.distributed.types import (
    DataType,
    EmbeddingEvent,
    ParameterSharding,
    ShardedModule,
    ShardingBucketMetadata,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import data_type_to_sparse_type
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.types import CopyMixIn

logger: logging.Logger = logging.getLogger(__name__)
_T = TypeVar("_T")

"""
torch.package safe functions from pyre_extensions. However, pyre_extensions is
not safe to use in code that will be torch.packaged, as it requires sys for
version checks
"""


def get_device_type() -> str:
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.mtia.is_available():
        device_type = "mtia"
    else:
        device_type = "cpu"
    return device_type


def get_class_name(obj: object) -> str:
    if obj is None:
        return "None"
    return obj.__class__.__name__


def assert_instance(obj: object, t: Type[_T]) -> _T:
    assert isinstance(obj, t), f"Got {get_class_name(obj)}"
    return obj


def none_throws(optional: Optional[_T], message: str = "Unexpected `None`") -> _T:
    """Convert an optional to its value. Raises an `AssertionError` if the
    value is `None`"""
    if optional is None:
        raise AssertionError(message)
    return optional


def append_prefix(prefix: str, name: str) -> str:
    """
    Appends provided prefix to provided name.
    """

    if prefix != "" and name != "":
        return prefix + "." + name
    else:
        return prefix + name


def filter_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]", name: str
) -> "OrderedDict[str, torch.Tensor]":
    """
    Filters state dict for keys that start with provided name.
    Strips provided name from beginning of key in the resulting state dict.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): input state dict to filter.
        name (str): name to filter from state dict keys.

    Returns:
        OrderedDict[str, torch.Tensor]: filtered state dict.
    """

    filtered_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(name + "."):
            # + 1 to length is to remove the '.' after the key
            filtered_state_dict[key[len(name) + 1 :]] = value
    return filtered_state_dict


def add_prefix_to_state_dict(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Adds prefix to all keys in state dict, in place.

    Args:
        state_dict (Dict[str, Any]): input state dict to update.
        prefix (str): name to filter from state dict keys.

    Returns:
        None.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        state_dict[prefix + key] = state_dict.pop(key)

    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            metadata[prefix + key] = metadata.pop(key)


def _get_unsharded_module_names_helper(
    model: torch.nn.Module,
    path: str,
    unsharded_module_names: Set[str],
) -> bool:
    sharded_children = set()
    for name, child in model.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _get_unsharded_module_names_helper(
                child,
                curr_path + ".",
                unsharded_module_names,
            )
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, _ in model.named_children():
            if name not in sharded_children:
                unsharded_module_names.add(path + name)

    return len(sharded_children) > 0


def get_unsharded_module_names(model: torch.nn.Module) -> List[str]:
    """
    Retrieves names of top level modules that do not contain any sharded sub-modules.

    Args:
        model (torch.nn.Module): model to retrieve unsharded module names from.

    Returns:
        List[str]: list of names of modules that don't have sharded sub-modules.
    """

    unsharded_module_names: Set[str] = set()
    _get_unsharded_module_names_helper(
        model,
        "",
        unsharded_module_names,
    )
    return list(unsharded_module_names)


class sharded_model_copy:
    """
    Allows copying of DistributedModelParallel module to a target device.

    Example::

        # Copying model to CPU.

        m = DistributedModelParallel(m)
        with sharded_model_copy("cpu"):
            m_cpu = copy.deepcopy(m)
    """

    def __init__(self, device: Optional[Union[str, int, torch.device]]) -> None:
        self.device = device

    def __enter__(self) -> None:
        self.t_copy_save_ = torch.Tensor.__deepcopy__
        self.p_copy_save_ = torch.nn.Parameter.__deepcopy__

        device = self.device

        def _tensor_copy(tensor, memo):
            if tensor.device != device:
                return tensor.detach().to(device)
            else:
                return tensor.detach().clone()

        def _no_copy(obj, memo):
            return obj

        _copy_or_not = _tensor_copy if self.device is not None else _no_copy

        def _param_copy(param, memo):
            return torch.nn.Parameter(
                _copy_or_not(param, memo), requires_grad=param.requires_grad
            )

        # pyrefly: ignore[bad-assignment]
        torch.Tensor.__deepcopy__ = _copy_or_not
        # pyrefly: ignore[bad-assignment]
        torch.nn.Parameter.__deepcopy__ = _param_copy
        # pyrefly: ignore[implicit-import]
        torch._C._distributed_c10d.ProcessGroupNCCL.__deepcopy__ = _no_copy
        # pyrefly: ignore[implicit-import]
        torch._C._distributed_c10d.ProcessGroupGloo.__deepcopy__ = _no_copy
        # pyrefly: ignore[implicit-import]
        torch._C._distributed_c10d.Work.__deepcopy__ = _no_copy
        torch.cuda.streams.Stream.__deepcopy__ = _no_copy

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        torch.Tensor.__deepcopy__ = self.t_copy_save_
        torch.nn.Parameter.__deepcopy__ = self.p_copy_save_
        # pyrefly: ignore[implicit-import]
        torch._C._distributed_c10d.ProcessGroupNCCL.__deepcopy__ = None
        # pyrefly: ignore[implicit-import]
        torch._C._distributed_c10d.ProcessGroupGloo.__deepcopy__ = None
        # pyrefly: ignore[implicit-import]
        torch._C._distributed_c10d.Work.__deepcopy__ = None
        torch.cuda.streams.Stream.__deepcopy__ = None


def copy_to_device(
    module: nn.Module,
    current_device: torch.device,
    to_device: torch.device,
) -> nn.Module:

    with sharded_model_copy(device=None):
        copy_module = copy.deepcopy(module)

    # Copy only weights with matching device.
    def _copy_if_device_match(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == current_device:
            return tensor.to(to_device)
        return tensor

    # if this is a sharded module, customize the copy
    if isinstance(copy_module, CopyMixIn):
        return copy_module.copy(to_device)
    copied_param = {
        name: torch.nn.Parameter(
            _copy_if_device_match(param.data), requires_grad=param.requires_grad
        )
        for name, param in copy_module.named_parameters(recurse=False)
    }
    copied_buffer = {
        name: _copy_if_device_match(buffer)
        for name, buffer in copy_module.named_buffers(recurse=False)
    }
    for name, param in copied_param.items():
        m = copy_module
        if "." in name:
            continue
        m.register_parameter(name, param)
    for name, buffer in copied_buffer.items():
        m = copy_module
        if "." in name:
            continue
        m.register_buffer(name, buffer)
    for child_name, child in copy_module.named_children():
        if not any([isinstance(submodule, CopyMixIn) for submodule in child.modules()]):
            child_copy = child._apply(_copy_if_device_match)
        else:
            child_copy = copy_to_device(child, current_device, to_device)
        copy_module.register_module(child_name, child_copy)
    return copy_module


class CopyableMixin(nn.Module):
    """
    Allows copying of module to a target device.

    Example::

        class MyModule(CopyableMixin):
            ...

    Args:
        device : torch.device to copy to

    Returns
        nn.Module on new device
    """

    def copy(
        self,
        device: torch.device,
    ) -> nn.Module:
        return copy_to_device(
            self,
            current_device=torch.device("cpu"),
            to_device=device,
        )


# Canonical mapping from torchrec optimizer classes to EmbOptimType.
_OPTIMIZER_CLASS_TO_EMB_OPT_TYPE: Dict[Type[torch.optim.Optimizer], EmbOptimType] = {
    # torch optimizers
    torch.optim.SGD: EmbOptimType.EXACT_SGD,
    torch.optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
    torch.optim.Adam: EmbOptimType.ADAM,
    # torchrec wrappers over these optims.
    # they accept an **unused kwargs portion, that let us set FBGEMM specific args such as
    # max gradient, etc
    trec_optim.SGD: EmbOptimType.EXACT_SGD,
    trec_optim.LarsSGD: EmbOptimType.LARS_SGD,
    trec_optim.LAMB: EmbOptimType.LAMB,
    trec_optim.PartialRowWiseLAMB: EmbOptimType.PARTIAL_ROWWISE_LAMB,
    trec_optim.Adam: EmbOptimType.ADAM,
    trec_optim.PartialRowWiseAdam: EmbOptimType.PARTIAL_ROWWISE_ADAM,
    trec_optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
    trec_optim.RowWiseAdagrad: EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}

# Inverse mapping from EmbOptimType to optimizer class.
# When multiple optimizer classes map to the same EmbOptimType, we prefer torchrec wrappers.
_EMB_OPT_TYPE_TO_OPTIMIZER_CLASS: Dict[EmbOptimType, Type[torch.optim.Optimizer]] = {
    EmbOptimType.EXACT_SGD: trec_optim.SGD,
    EmbOptimType.EXACT_ADAGRAD: trec_optim.Adagrad,
    EmbOptimType.ADAM: trec_optim.Adam,
    EmbOptimType.LARS_SGD: trec_optim.LarsSGD,
    EmbOptimType.LAMB: trec_optim.LAMB,
    EmbOptimType.PARTIAL_ROWWISE_LAMB: trec_optim.PartialRowWiseLAMB,
    EmbOptimType.PARTIAL_ROWWISE_ADAM: trec_optim.PartialRowWiseAdam,
    EmbOptimType.EXACT_ROWWISE_ADAGRAD: trec_optim.RowWiseAdagrad,
}


def optimizer_type_to_emb_opt_type(
    optimizer_class: Type[torch.optim.Optimizer],
) -> Optional[EmbOptimType]:
    """
    Convert a torch.optim.Optimizer class to its corresponding EmbOptimType.

    Args:
        optimizer_class: The optimizer class to convert.

    Returns:
        The corresponding EmbOptimType.

    Raises:
        ValueError: If the optimizer class is not in the mapping.
    """
    # TODO add more optimizers to be in parity with ones provided by FBGEMM
    # TODO kwargs accepted by fbgemm and and canonical optimizers are different
    # may need to add special handling for them
    if optimizer_class not in _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE:
        raise ValueError(f"Cannot cast {optimizer_class} to an EmbOptimType")
    return _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE[optimizer_class]


def emb_opt_type_to_optimizer_class(
    emb_opt_type: Optional[EmbOptimType],
) -> Optional[Type[torch.optim.Optimizer]]:
    """
    Convert EmbOptimType to torch.optim.Optimizer class for optimizer storage calculation.

    This is the inverse of optimizer_type_to_emb_opt_type. When multiple optimizer classes
    map to the same EmbOptimType, this function returns the torchrec wrapper class.

    Args:
        emb_opt_type: The EmbOptimType to convert, or None.

    Returns:
        The corresponding optimizer class, or None if emb_opt_type is None or not found.
    """
    if emb_opt_type is None:
        return None
    return _EMB_OPT_TYPE_TO_OPTIMIZER_CLASS.get(emb_opt_type)


def merge_fused_params(
    fused_params: Optional[Dict[str, Any]] = None,
    param_fused_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Configure the fused_params including cache_precision if the value is not preset.

    Values set in table_level_fused_params take precidence over the global fused_params

    Args:
        fused_params (Optional[Dict[str, Any]]): the original fused_params
        grouped_fused_params

    Returns:
        [Dict[str, Any]]: a non-null configured fused_params dictionary to be
        used to configure the embedding lookup kernel
    """

    if fused_params is None:
        fused_params = {}
    if param_fused_params is None:
        param_fused_params = {}
    if "lr" in param_fused_params:
        param_fused_params["learning_rate"] = param_fused_params.pop("lr")

    _fused_params = copy.deepcopy(fused_params)
    _fused_params.update(param_fused_params)
    return _fused_params


def add_params_from_parameter_sharding(
    fused_params: Optional[Dict[str, Any]],
    parameter_sharding: ParameterSharding,
) -> Dict[str, Any]:
    """
    Extract params from parameter sharding and then add them to fused_params.

    Params from parameter sharding will override the ones in fused_params if they
    exist already.

    Args:
        fused_params (Optional[Dict[str, Any]]): the existing fused_params
        parameter_sharding (ParameterSharding): the parameter sharding to use

    Returns:
        [Dict[str, Any]]: the fused_params dictionary with params from parameter
        sharding added.

    """
    if fused_params is None:
        fused_params = {}

    # update fused_params using params from parameter_sharding
    # this will take precidence over the fused_params provided from sharders
    if parameter_sharding.cache_params is not None:
        cache_params = parameter_sharding.cache_params
        if cache_params.algorithm is not None:
            fused_params["cache_algorithm"] = cache_params.algorithm
        if cache_params.load_factor is not None:
            fused_params["cache_load_factor"] = cache_params.load_factor
        if cache_params.reserved_memory is not None:
            fused_params["cache_reserved_memory"] = cache_params.reserved_memory
        if cache_params.precision is not None:
            fused_params["cache_precision"] = cache_params.precision
        if cache_params.prefetch_pipeline is not None:
            fused_params["prefetch_pipeline"] = cache_params.prefetch_pipeline
        if cache_params.multipass_prefetch_config is not None:
            fused_params["multipass_prefetch_config"] = (
                cache_params.multipass_prefetch_config
            )

    if parameter_sharding.enforce_hbm is not None:
        fused_params["enforce_hbm"] = parameter_sharding.enforce_hbm

    if parameter_sharding.stochastic_rounding is not None:
        fused_params["stochastic_rounding"] = parameter_sharding.stochastic_rounding

    if parameter_sharding.bounds_check_mode is not None:
        fused_params["bounds_check_mode"] = parameter_sharding.bounds_check_mode

    if parameter_sharding.output_dtype is not None:
        fused_params["output_dtype"] = parameter_sharding.output_dtype

    if (
        parameter_sharding.compute_kernel
        in {
            EmbeddingComputeKernel.KEY_VALUE.value,
            EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
            EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
        }
        and parameter_sharding.key_value_params is not None
    ):
        kv_params = parameter_sharding.key_value_params
        key_value_params_dict = asdict(kv_params)
        key_value_params_dict = {
            k: v for k, v in key_value_params_dict.items() if v is not None
        }
        if kv_params.stats_reporter_config:
            key_value_params_dict["stats_reporter_config"] = (
                kv_params.stats_reporter_config
            )
        fused_params.update(key_value_params_dict)

    # print warning if sharding_type is data_parallel or kernel is dense
    if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
        logger.warning(
            f"Sharding Type is {parameter_sharding.sharding_type}, "
            "caching params will be ignored"
        )
    elif parameter_sharding.compute_kernel == EmbeddingComputeKernel.DENSE.value:
        logger.warning(
            f"Compute Kernel is {parameter_sharding.compute_kernel}, "
            "caching params will be ignored"
        )

    # calling `get_additional_fused_params` for customized kernel
    # it will be updated to the `fused_params` dict
    if hasattr(
        parameter_sharding, "get_additional_fused_params"
    ) and parameter_sharding.compute_kernel in {
        EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
    }:
        # type: ignore[attr-defined]
        fused_params.update(parameter_sharding.get_additional_fused_params())

    return fused_params


def convert_to_fbgemm_types(fused_params: Dict[str, Any]) -> Dict[str, Any]:
    if "cache_precision" in fused_params:
        if isinstance(fused_params["cache_precision"], DataType):
            fused_params["cache_precision"] = data_type_to_sparse_type(
                fused_params["cache_precision"]
            )

    if "weights_precision" in fused_params:
        if isinstance(fused_params["weights_precision"], DataType):
            fused_params["weights_precision"] = data_type_to_sparse_type(
                fused_params["weights_precision"]
            )

    if "output_dtype" in fused_params:
        if isinstance(fused_params["output_dtype"], DataType):
            fused_params["output_dtype"] = data_type_to_sparse_type(
                fused_params["output_dtype"]
            )

    return fused_params


def init_parameters(module: nn.Module, device: torch.device) -> None:
    with torch.no_grad():
        has_meta_param = any(t.is_meta for t in module.parameters())
        not_on_target_device = any(t.device != device for t in module.parameters())
        if not_on_target_device:
            module.to_empty(device=device) if has_meta_param else module.to(device)

            def maybe_reset_parameters(m: nn.Module) -> None:
                if hasattr(m, "reset_parameters"):
                    # pyrefly: ignore[not-callable]
                    m.reset_parameters()

            module.apply(maybe_reset_parameters)


def maybe_annotate_embedding_event(
    event: EmbeddingEvent,
    module_fqn: Optional[str],
    sharding_type: Optional[str],
    #  received 1.
) -> AbstractContextManager[None]:
    if module_fqn and sharding_type:
        annotation = f"[{event.value}]_[{module_fqn}]_[{sharding_type}]"
        return record_function(annotation)
    else:
        return nullcontext()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child.
    Useful in debugging multiprocessed code

    Example::

        from torchrec.multiprocessing_utils import ForkedPdb

        if dist.get_rank() == 0:
            ForkedPdb().set_trace()
        dist.barrier()
    """

    def interaction(self, *args, **kwargs) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")  # noqa
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def create_global_tensor_shape_stride_from_metadata(
    parameter_sharding: ParameterSharding, devices_per_node: Optional[int] = None
) -> Tuple[torch.Size, Tuple[int, int]]:
    """
    Create a global tensor shape and stride from shard metadata.

    Returns:
        torch.Size: global tensor shape.
        tuple: global tensor stride.
    """
    size = None
    if parameter_sharding.sharding_type == ShardingType.COLUMN_WISE.value:
        # pyrefly: ignore[missing-attribute]
        row_dim = parameter_sharding.sharding_spec.shards[0].shard_sizes[0]
        col_dim = 0
        # pyrefly: ignore[missing-attribute]
        for shard in parameter_sharding.sharding_spec.shards:
            col_dim += shard.shard_sizes[1]
        size = torch.Size([row_dim, col_dim])
    elif (
        parameter_sharding.sharding_type == ShardingType.ROW_WISE.value
        or parameter_sharding.sharding_type == ShardingType.TABLE_ROW_WISE.value
    ):
        row_dim = 0
        # pyrefly: ignore[missing-attribute]
        col_dim = parameter_sharding.sharding_spec.shards[0].shard_sizes[1]
        # pyrefly: ignore[missing-attribute]
        for shard in parameter_sharding.sharding_spec.shards:
            row_dim += shard.shard_sizes[0]
        size = torch.Size([row_dim, col_dim])
    elif parameter_sharding.sharding_type == ShardingType.TABLE_WISE.value:
        # pyrefly: ignore[missing-attribute]
        size = torch.Size(parameter_sharding.sharding_spec.shards[0].shard_sizes)
    elif parameter_sharding.sharding_type == ShardingType.GRID_SHARD.value:
        # we need node group size to appropriately calculate global shape from shard
        assert devices_per_node is not None
        row_dim, col_dim = 0, 0
        # pyrefly: ignore[missing-attribute]
        num_cw_shards = len(parameter_sharding.sharding_spec.shards) // devices_per_node
        for _ in range(num_cw_shards):
            # pyrefly: ignore[missing-attribute]
            col_dim += parameter_sharding.sharding_spec.shards[0].shard_sizes[1]
        for _ in range(devices_per_node):
            # pyrefly: ignore[missing-attribute]
            row_dim += parameter_sharding.sharding_spec.shards[0].shard_sizes[0]
        size = torch.Size([row_dim, col_dim])
    # pyrefly: ignore[bad-return]
    return size, (size[1], 1) if size else (torch.Size([0, 0]), (0, 1))


def get_bucket_metadata_from_shard_metadata(
    shards: List[ShardMetadata],
    num_buckets: int,
) -> ShardingBucketMetadata:
    """
    Calculate the bucket metadata from shard metadata.

    This function assumes the table is to be row-wise sharded in equal sized buckets across bucket boundaries.
    It computes the number of buckets per shard and the bucket size.

    Args:
        shards (List[ShardMetadata]): Shard metadata for all shards of a table.
        num_buckets (int): The number of buckets to divide the table into.

    Returns:
        ShardingBucketMetadata: An object containing the number of buckets per shard and the bucket size.
    """
    assert len(shards) > 0, "Shards cannot be empty"
    table_size = shards[-1].shard_offsets[0] + shards[-1].shard_sizes[0]
    assert (
        table_size % num_buckets == 0
    ), f"Table size '{table_size}' must be divisible by num_buckets '{num_buckets}'"
    bucket_size = table_size // num_buckets
    bucket_metadata: ShardingBucketMetadata = ShardingBucketMetadata(
        num_buckets_per_shard=[], bucket_offsets_per_shard=[], bucket_size=bucket_size
    )
    current_bucket_offset = 0
    for shard in shards:
        assert (
            len(shard.shard_offsets) == 1 or shard.shard_offsets[1] == 0
        ), f"Shard shard_offsets[1] '{shard.shard_offsets[1]}' is not 0. Table should be only row-wise sharded for bucketization"
        assert (
            shard.shard_sizes[0] % bucket_size == 0
        ), f"Shard size[0] '{shard.shard_sizes[0]}' is not divisible by bucket size '{bucket_size}'"
        num_buckets_in_shard = shard.shard_sizes[0] // bucket_size
        bucket_metadata.num_buckets_per_shard.append(num_buckets_in_shard)
        bucket_metadata.bucket_offsets_per_shard.append(current_bucket_offset)
        current_bucket_offset += num_buckets_in_shard

    return bucket_metadata


def _group_sharded_modules(
    module: nn.Module,
) -> List[torch.nn.Module]:
    # Post init DMP, save the embedding kernels
    sharded_modules: List[torch.nn.Module] = []

    def _find_sharded_modules(
        module: torch.nn.Module,
    ) -> None:
        if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
            sharded_modules.append(module)
        if hasattr(module, "_lookups"):
            #  not a function.
            # pyrefly: ignore[not-iterable]
            for lookup in module._lookups:
                _find_sharded_modules(lookup)
            return
        for _, child in module.named_children():
            _find_sharded_modules(child)

    _find_sharded_modules(module)
    return sharded_modules


def _convert_weights(
    weights: torch.Tensor,
    converted_dtype: SparseType,
) -> torch.Tensor:
    torch_dtype = converted_dtype.as_dtype()
    new_weights = weights.to(dtype=torch_dtype)
    weights.untyped_storage().resize_(0)
    return new_weights


def weights_bytes_in_emb_kernel(emb: nn.Module) -> int:
    total_bytes = (
        # pyrefly: ignore[not-callable]
        emb.weights_dev.element_size() * emb.weights_dev.numel()
        # pyrefly: ignore[not-callable]
        + emb.weights_host.element_size() * emb.weights_host.numel()
        # pyrefly: ignore[not-callable]
        + emb.weights_uvm.element_size() * emb.weights_uvm.numel()
    )
    return total_bytes


class EmbeddingQuantizationUtils:
    def __init__(self) -> None:
        self._emb_kernel_to_sparse_dtype: Dict[
            SplitTableBatchedEmbeddingBagsCodegen, SparseType
        ] = {}

    def quantize_embedding_modules(
        self, module: nn.Module, converted_dtype: DataType
    ) -> None:
        sharded_embs = _group_sharded_modules(module)
        sharded_embs.sort(key=weights_bytes_in_emb_kernel)
        logger.info(
            f"[TorchRec] Converting embedding modules to converted_dtype={converted_dtype.value} quantization"
        )
        converted_sparse_dtype = data_type_to_sparse_type(converted_dtype)

        for emb_kernel in sharded_embs:
            emb_kernel.weights_dev = _convert_weights(
                # pyrefly: ignore[bad-argument-type]
                emb_kernel.weights_dev,
                converted_sparse_dtype,
            )
            emb_kernel.weights_host = _convert_weights(
                # pyrefly: ignore[bad-argument-type]
                emb_kernel.weights_host,
                converted_sparse_dtype,
            )
            emb_kernel.weights_uvm = _convert_weights(
                # pyrefly: ignore[bad-argument-type]
                emb_kernel.weights_uvm,
                converted_sparse_dtype,
            )
            # pyrefly: ignore[no-matching-overload]
            self._emb_kernel_to_sparse_dtype.setdefault(
                emb_kernel, emb_kernel.weights_precision
            )

            # pyrefly: ignore[bad-argument-type]
            emb_kernel.weights_precision = converted_sparse_dtype

    def recreate_embedding_modules(
        self,
        module: nn.Module,
    ) -> None:
        sharded_embs = _group_sharded_modules(module)
        sharded_embs.sort(key=weights_bytes_in_emb_kernel)

        for emb_kernel in sharded_embs:
            # pyrefly: ignore[bad-index]
            converted_sparse_dtype = self._emb_kernel_to_sparse_dtype[emb_kernel]

            emb_kernel.weights_dev = _convert_weights(
                # pyrefly: ignore[bad-argument-type]
                emb_kernel.weights_dev,
                converted_sparse_dtype,
            )
            emb_kernel.weights_host = _convert_weights(
                # pyrefly: ignore[bad-argument-type]
                emb_kernel.weights_host,
                converted_sparse_dtype,
            )
            emb_kernel.weights_uvm = _convert_weights(
                # pyrefly: ignore[bad-argument-type]
                emb_kernel.weights_uvm,
                converted_sparse_dtype,
            )
        self._recalculate_torch_state(module)

    def _recalculate_torch_state(self, module: nn.Module) -> None:
        def _recalculate_torch_state_helper(
            module: torch.nn.Module,
        ) -> None:
            if hasattr(module, "_lookups") or hasattr(module, "_lookup"):
                #  not a function.
                # pyrefly: ignore[not-callable]
                module._initialize_torch_state(skip_registering=True)
                return
            for _, child in module.named_children():
                _recalculate_torch_state_helper(child)

        _recalculate_torch_state_helper(module)


def modify_input_for_feature_processor(
    features: KeyedJaggedTensor,
    feature_processors: Union[nn.ModuleDict, FeatureProcessorsCollection],
    is_collection: bool,
) -> None:
    """
    This function applies the feature processor pre input dist. This way we
    can support row wise based sharding mechanisms.

    This is an inplace modifcation of the input KJT.
    """
    with torch.no_grad():
        if features.weights_or_none() is None:
            # force creation of weights, this way the feature jagged tensor weights are tied to the original KJT
            features._weights = torch.zeros_like(features.values(), dtype=torch.float32)

        if is_collection:
            if hasattr(feature_processors, "pre_process_input"):
                # pyrefly: ignore[not-callable]
                feature_processors.pre_process_input(features)
            else:
                logging.info(
                    f"[Feature Processor Pipeline] Skipping pre_process_input for feature processor {feature_processors=}"
                )
        else:
            # per feature process
            for feature in features.keys():
                # pyrefly: ignore[unsupported-operation]
                if feature in feature_processors:
                    # pyrefly: ignore[bad-index]
                    feature_processor = feature_processors[feature]
                    if hasattr(feature_processor, "pre_process_input"):
                        # pyrefly: ignore[not-callable]
                        feature_processor.pre_process_input(features[feature])
                    else:
                        logging.info(
                            f"[Feature Processor Pipeline] Skipping pre_process_input for feature processor {feature_processor=}"
                        )
                else:
                    features[feature].weights().copy_(
                        torch.ones(
                            features[feature].values().shape[0],
                            dtype=torch.float32,
                            device=features[feature].values().device,
                        )
                    )


def _collect_cuda_tensors_from_value(
    value: Any,
    min_size_bytes: int = 1024 * 1024,  # 1MB default threshold
) -> List[torch.Tensor]:
    """
    Recursively collect CUDA tensors from a value.

    Handles nested structures like:
    - Direct tensors
    - Tuples/lists of tensors (e.g., Shampoo's factor_matrices)
    - Objects with tensor attributes (e.g., ShampooKroneckerFactors)
    - Nested dicts

    Args:
        value: The value to extract tensors from.
        min_size_bytes: Minimum tensor size in bytes to include. Tensors smaller
            than this threshold are skipped to avoid overhead of stashing small
            tensors. Default is 1MB.

    Returns:
        List of CUDA tensors found in the value that meet the size threshold.
    """
    tensors: List[torch.Tensor] = []

    if isinstance(value, torch.Tensor):
        if value.is_cuda and value.numel() > 0:
            tensor_size_bytes = value.numel() * value.element_size()
            if tensor_size_bytes >= min_size_bytes:
                tensors.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(_collect_cuda_tensors_from_value(item, min_size_bytes))
    elif isinstance(value, dict):
        for v in value.values():
            tensors.extend(_collect_cuda_tensors_from_value(v, min_size_bytes))
    elif hasattr(value, "__dataclass_fields__"):
        # Handle dataclass-like objects (e.g., ShampooKroneckerFactors)
        for field_name in value.__dataclass_fields__:
            field_value = getattr(value, field_name, None)
            if field_value is not None:
                tensors.extend(
                    _collect_cuda_tensors_from_value(field_value, min_size_bytes)
                )
    elif hasattr(value, "__dict__"):
        # Handle generic objects with attributes
        for attr_value in value.__dict__.values():
            tensors.extend(_collect_cuda_tensors_from_value(attr_value, min_size_bytes))

    return tensors


def stash_optimizer_state(
    optimizer: torch.optim.Optimizer,
    stash_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[
    Callable[[], None],
    Callable[[Optional[torch.Tensor]], None],
    Callable[[Optional[torch.Tensor]], None],
]:
    """
    Stash optimizer state tensors from HBM to CPU asynchronously.

    This function immediately starts an async copy of optimizer state tensors from GPU (HBM)
    to CPU (pinned memory), and returns three callback functions for managing the
    stash/restore lifecycle. This is useful for reducing HBM memory usage during
    training when optimizer state is not needed (e.g., during forward/backward pass).

    This function works with any optimizer that stores state tensors in `optimizer.state`,
    including Shampoo (with nested ShampooKroneckerFactors), Adam, SGD with momentum, etc.

    Args:
        optimizer: A PyTorch optimizer containing state tensors to stash.
        stash_stream: Optional CUDA stream for async HBM→CPU transfer. If None,
            a new stream will be created.

    Returns:
        A tuple of three callback functions:
        - free_hbm: Frees HBM storage from CPU side (call after stash copy completes)
        - restore: Retrieves stashed data from CPU back to HBM asynchronously
        - await_restore: Pauses current stream awaiting restore completion

    Usage:
        >>> # After forward pass completes (optimizer state not needed):
        >>> free_hbm, restore, await_restore = stash_optimizer_state(optimizer)
        >>> # ... do forward/backward while stash copy is in progress ...
        >>> free_hbm()  # Free HBM once stash copy completes
        >>> # ... HBM is now free for forward/backward pass ...
        >>> # Before optimizer.step():
        >>> restore()  # Start async restore from CPU to HBM
        >>> await_restore()  # Wait for restore to complete before using state
        >>> optimizer.step()  # Now optimizer state is available

        >>> # With custom stream:
        >>> stream = torch.cuda.Stream()
        >>> free_hbm, restore, await_restore = stash_optimizer_state(optimizer, stash_stream=stream)

    Note:
        - Uses pinned CPU memory for efficient async transfers
        - Uses separate CUDA stream to overlap with computation
        - free_hbm blocks CPU until stash copy completes (necessary for safe resize)
        - restore starts async copy, await_restore blocks GPU stream until complete
        - Only CUDA tensors are stashed; CPU tensors and non-tensor state are left unchanged
        - Supports nested structures like Shampoo's ShampooKroneckerFactors
    """
    # (state_tensor ref, cpu_buffer, cuda_stream, stash_event)
    stash_data: List[
        Tuple[torch.Tensor, torch.Tensor, torch.cuda.Stream, torch.cuda.Event]
    ] = []

    # List to store restore events for wait_for_restore
    restore_events: List[torch.cuda.Event] = []

    # Iterate through optimizer state and stash CUDA tensors
    for _param, state_dict in optimizer.state.items():
        if not isinstance(state_dict, dict):
            continue

        for _state_key, state_value in state_dict.items():
            # Recursively collect all CUDA tensors from the state value
            # This handles nested structures like Shampoo's ShampooKroneckerFactors
            cuda_tensors = _collect_cuda_tensors_from_value(state_value)

            for tensor in cuda_tensors:
                # Create pinned CPU buffer for efficient async DMA transfer
                cpu_buffer = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )

                # Use provided stream or create a new one for this transfer
                stream = stash_stream or torch.cuda.Stream(device=tensor.device)

                # Ensure all operations on the default stream complete before we start copying
                stream.wait_stream(torch.cuda.current_stream())

                # Start async copy from HBM to CPU
                size_mb = tensor.numel() * tensor.element_size() / (1024**2)
                with record_function(
                    f"stash optimizer state to host ({size_mb:.2f} MB)"
                ):
                    with torch.cuda.stream(stream):
                        cpu_buffer.copy_(tensor, non_blocking=True)
                        stash_event = torch.cuda.Event()
                        stash_event.record(stream)

                stash_data.append((tensor, cpu_buffer, stream, stash_event))

    def free_hbm() -> None:
        """Free HBM storage from CPU side after stash copy completes."""
        for state_tensor, _cpu_buffer, _stream, stash_event in stash_data:
            # CPU-blocking wait: block CPU thread until GPU copy completes
            stash_event.synchronize()
            # Free HBM storage
            state_tensor.untyped_storage().resize_(0)

    def restore(_grad: Optional[torch.Tensor] = None) -> None:
        """Restore state tensors from CPU to HBM asynchronously."""
        restore_events.clear()
        for hbm_ref, cpu_buffer, stream, _stash_event in stash_data:
            # Re-allocate HBM storage
            storage_size = cpu_buffer.numel() * cpu_buffer.element_size()
            hbm_ref.untyped_storage().resize_(storage_size)

            # Copy data back to HBM using a temporary tensor to bypass autograd
            size_mb = cpu_buffer.numel() * cpu_buffer.element_size() / (1024**2)
            stream.wait_stream(torch.cuda.current_stream())
            with record_function(
                f"restore optimizer state from host ({size_mb:.2f} MB)"
            ):
                with torch.cuda.stream(stream):
                    tmp = torch.tensor([], dtype=hbm_ref.dtype, device=hbm_ref.device)
                    tmp.set_(
                        hbm_ref.untyped_storage(),
                        storage_offset=0,
                        size=hbm_ref.shape,
                        stride=hbm_ref.stride(),
                    )
                    tmp.copy_(cpu_buffer, non_blocking=True)
                    restore_event = torch.cuda.Event()
                    restore_event.record(stream)
                    restore_events.append(restore_event)

    def await_restore(_grad: Optional[torch.Tensor] = None) -> None:
        """Pause current stream awaiting restore completion."""
        for restore_event in restore_events:
            torch.cuda.current_stream().wait_event(restore_event)

    return free_hbm, restore, await_restore


def stash_embedding_weights(
    lookup: nn.Module,
    stash_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[
    Callable[[], None],
    Callable[[Optional[torch.Tensor]], None],
    Callable[[Optional[torch.Tensor]], None],
]:
    """
    Stash embedding weights from HBM to CPU asynchronously.

    This function immediately starts an async copy of embedding weights from GPU (HBM)
    to CPU (pinned memory), and returns three callback functions for managing the
    stash/restore lifecycle.

    Args:
        lookup: A lookup module (e.g., GroupedPooledEmbeddingsLookup) containing
            embedding modules with weights to stash.
        stash_stream: Optional CUDA stream for async HBM→CPU transfer. If None,
            a new stream will be created.

    Returns:
        A tuple of three callback functions:
        - free_hbm: Frees HBM storage from CPU side (call after stash copy completes)
        - restore: Retrieves stashed data from CPU back to HBM asynchronously
        - await_restore: Pauses current stream awaiting restore completion

    Usage:
        >>> # After forward lookup completes:
        >>> free_hbm, restore, await_restore = stash_embedding_weights(lookup)
        >>> # ... do other work while stash copy is in progress ...
        >>> free_hbm()  # Free HBM once stash copy completes
        >>> # ... HBM is now free for other ops ...
        >>> # Before backward (or next forward):
        >>> restore()  # Start async restore from CPU to HBM
        >>> await_restore()  # Wait for restore to complete before using weights

        >>> # With custom stream:
        >>> stream = torch.cuda.Stream()
        >>> free_hbm, restore, await_restore = stash_embedding_weights(lookup, stash_stream=stream)

    Note:
        - Uses pinned CPU memory for efficient async transfers
        - Uses separate CUDA stream to overlap with computation
        - free_hbm blocks CPU until stash copy completes (necessary for safe resize)
        - restore starts async copy, await_restore blocks GPU stream until complete
    """
    # Handle DDP wrapper - unwrap to get the actual module
    module = lookup.module if hasattr(lookup, "module") else lookup

    # Early return if module doesn't have embedding modules
    if not hasattr(module, "_emb_modules"):
        return lambda: None, lambda _grad: None, lambda _grad: None

    # List to store stash data for each embedding module:
    # (weights_dev tensor ref, cpu_buffer, cuda_stream, stash_event)
    stash_data: List[
        Tuple[torch.Tensor, torch.Tensor, torch.cuda.Stream, torch.cuda.Event]
    ] = []

    # List to store restore events for wait_for_restore
    restore_events: List[torch.cuda.Event] = []

    # Process each embedding module and start async HBM → CPU copy
    for emb_module in module._emb_modules:
        # Skip if no inner embedding module (e.g., BatchedFusedEmbeddingBag)
        if not hasattr(emb_module, "_emb_module"):
            continue

        # Get the inner FBGEMM TBE (SplitTableBatchedEmbeddingBagsCodegen)
        inner = emb_module._emb_module
        if not hasattr(inner, "weights_dev"):
            continue

        # weights_dev is the actual embedding table weights on GPU
        weights_dev = inner.weights_dev
        if weights_dev is None or not weights_dev.is_cuda:
            continue

        # Create pinned CPU buffer for efficient async DMA transfer
        # Pinned memory allows GPU to directly access CPU memory via DMA
        cpu_buffer = torch.empty(
            weights_dev.shape,
            dtype=weights_dev.dtype,
            device="cpu",
            pin_memory=True,
        )

        # Use provided stream or create a new one for this transfer
        stream = stash_stream or torch.cuda.Stream(device=weights_dev.device)

        # Ensure all operations on the default stream complete before we start copying
        # This prevents reading weights while they're still being written by forward pass
        stream.wait_stream(torch.cuda.current_stream())

        # Start async copy from HBM to CPU
        size_gb = weights_dev.numel() * weights_dev.element_size() / (1024**3)
        with record_function(f"stash embedding to host ({size_gb:.2f} GB)"):
            with torch.cuda.stream(stream):
                # non_blocking=True: CPU returns immediately, copy runs async on GPU
                cpu_buffer.copy_(weights_dev, non_blocking=True)
                # Record event to mark when copy completes
                stash_event = torch.cuda.Event()
                stash_event.record(stream)

        stash_data.append((weights_dev, cpu_buffer, stream, stash_event))

    def free_hbm() -> None:
        """Free HBM storage from CPU side after stash copy completes."""
        for weights_dev, _cpu_buffer, _stream, stash_event in stash_data:
            # CPU-blocking wait: block CPU thread until GPU copy completes
            # This is necessary because resize_(0) is a CPU operation that frees GPU memory
            # We must ensure data is safely on CPU before freeing HBM
            stash_event.synchronize()

            # Free HBM storage immediately after copy completes
            # This makes the GPU memory available for other operations
            # caveat: this is a CPU-side operation
            weights_dev.untyped_storage().resize_(0)

    def restore(_grad: Optional[torch.Tensor] = None) -> None:
        """Restore weights from CPU to HBM asynchronously."""
        restore_events.clear()
        for hbm_ref, cpu_buffer, stream, _stash_event in stash_data:
            # Re-allocate HBM storage (was freed after stash with resize_(0))
            storage_size = cpu_buffer.numel() * cpu_buffer.element_size()
            hbm_ref.untyped_storage().resize_(storage_size)

            # Copy data back to HBM using a temporary tensor to bypass autograd
            # Problem: copy_() on the original tensor increments its version counter,
            # causing "modified by inplace operation" error during backward
            # Solution: create a new tensor viewing the same storage - it has its own
            # version counter, so copy_() on it won't affect the original tensor's version
            size_gb = cpu_buffer.numel() * cpu_buffer.element_size() / (1024**3)
            stream.wait_stream(torch.cuda.current_stream())
            with record_function(f"restore embedding from host ({size_gb:.2f} GB)"):
                with torch.cuda.stream(stream):
                    # Create a temporary tensor that views the same storage
                    # but is not tracked by autograd (fresh version counter)
                    tmp = torch.tensor([], dtype=hbm_ref.dtype, device=hbm_ref.device)
                    tmp.set_(
                        hbm_ref.untyped_storage(),
                        storage_offset=0,
                        size=hbm_ref.shape,
                        stride=hbm_ref.stride(),
                    )
                    # Async copy from CPU to HBM
                    tmp.copy_(cpu_buffer, non_blocking=True)
                    # Record event to mark when copy completes
                    restore_event = torch.cuda.Event()
                    restore_event.record(stream)
                    restore_events.append(restore_event)

    def await_restore(_grad: Optional[torch.Tensor] = None) -> None:
        """Pause current stream awaiting restore completion."""
        # Make the default stream wait for all restore operations to complete
        # This is GPU-stream blocking (not CPU-blocking), allowing CPU to continue
        # while ensuring backward pass (on default stream) waits for restore
        for restore_event in restore_events:
            torch.cuda.current_stream().wait_event(restore_event)

    return free_hbm, restore, await_restore
