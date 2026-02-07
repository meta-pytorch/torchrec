#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Union

import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor


def state_dict_gather(
    src: Dict[str, Union[torch.Tensor, ShardedTensor]],
    dst: Dict[str, torch.Tensor],
) -> None:
    """
    Gathers the values of the src state_dict of the keys present in the dst state_dict. Can handle ShardedTensors in the src state_dict.

    Args:
        src (Dict[str, Union[torch.Tensor, ShardedTensor]]): source's state_dict for this rank
        dst (Dict[str, torch.Tensor]): destination's state_dict
    """
    for key, dst_tensor in dst.items():
        src_tensor = src[key]
        if isinstance(src_tensor, ShardedTensor):
            src_tensor.gather(
                out=dst_tensor if (dist.get_rank() == 0) else None,
                dtype=dst_tensor.dtype,
            )
        elif isinstance(src_tensor, torch.Tensor):
            dst_tensor.copy_(src_tensor)
        else:
            raise ValueError(f"Unsupported tensor {key} type {type(src_tensor)}")


def state_dict_all_gather_keys(
    state_dict: Dict[str, Union[torch.Tensor, ShardedTensor]],
    pg: ProcessGroup,
) -> List[str]:
    """
    Gathers all the keys of the state_dict from all ranks. Can handle ShardedTensors in the state_dict.

    Args:
        state_dict (Dict[str, Union[torch.Tensor, ShardedTensor]]): keys of this state_dict will be gathered
        pg (ProcessGroup): Process Group used for comms
    """
    names = list(state_dict.keys())
    all_names = [None] * dist.get_world_size(pg)
    dist.all_gather_object(all_names, names, pg)
    deduped_names = set()
    for local_names in all_names:
        for name in local_names:
            deduped_names.add(name)
    return sorted(deduped_names)


def state_dict_to_device(
    state_dict: Dict[str, Union[torch.Tensor, ShardedTensor]],
    pg: ProcessGroup,
    device: torch.device,
) -> Dict[str, Union[torch.Tensor, ShardedTensor]]:
    """
    Moves a state_dict to a device with a process group. Can handle ShardedTensors in the state_dict.

    Args:
        state_dict (Dict[str, Union[torch.Tensor, ShardedTensor]]): state_dict to move
        pg (ProcessGroup): Process Group used for comms
        device (torch.device): device to put state_dict on
    """
    ret = {}
    all_keys = state_dict_all_gather_keys(state_dict, pg)
    for key in all_keys:
        if key in state_dict:
            tensor = state_dict[key]
            if isinstance(tensor, ShardedTensor):
                copied_shards = [
                    Shard.from_tensor_and_offsets(
                        tensor=shard.tensor.to(device),
                        shard_offsets=shard.metadata.shard_offsets,
                        rank=dist.get_rank(pg),
                    )
                    for shard in tensor.local_shards()
                ]
                ret[key] = ShardedTensor._init_from_local_shards(
                    copied_shards,
                    tensor.metadata().size,
                    process_group=pg,
                )
            elif isinstance(tensor, torch.Tensor):
                ret[key] = tensor.to(device)
            else:
                raise ValueError(f"Unsupported tensor {key} type {type(tensor)}")
        else:
            # No state_dict entries for table-wise sharding,
            # but need to follow full-sync.
            ret[key] = ShardedTensor._init_from_local_shards(
                [],
                [],
                process_group=pg,
            )
    return ret
