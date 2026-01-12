#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copyreg
import io
import os
import pickle
import uuid
from typing import cast, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
import torchrec
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn
from torch.multiprocessing.reductions import (
    reduce_storage,
    reduce_typed_storage,
    reduce_typed_storage_child,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection


gloo_pg: Optional[dist.ProcessGroup] = None


def share_tensor_via_shm(
    tensor: Optional[torch.Tensor], src_rank: int = 0
) -> torch.Tensor:
    """
    Share a tensor via shared memory with local peers.
    This is a collective function that must be called by all processes within
    the global process group. Rank `src_rank` must pass in the tensor it wants
    to share.
    NOTE: this is a simple implementation that only supports the single-host,
    multi-process environment. Multi-host support is possible but is slightly
    more complicated.

    See [`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html) and
    [best practices](https://pytorch.org/docs/1.6.0/notes/multiprocessing.html?highlight=multiprocessing) for more information on shared memory

    Args:
        tensor: The tensor to share.
        src_rank: The rank sharing the tensor.
    Returns:
        The tensor shared via shared memory.

    Example::

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.barrier()

        if dist.get_rank() == 0:
            # Pretend that we are loading the pretrained embedding weight from a parquet file on rank 0.
            emb = torch.rand(2000000, 64)
            # Share the tensor to local peers via shared memory
            emb = share_tensor_via_shm(tensor=emb)
        else:
            # Received the tensor shared by rank 0 via shared memory
            emb = share_tensor_via_shm(tensor=None)

        assert emb.is_shared()
    """

    if not dist.is_initialized():
        raise RuntimeError("Global process group is not initialized")

    global gloo_pg
    if gloo_pg is None:
        if dist.get_backend() == "gloo":
            gloo_pg = dist.group.WORLD
        else:
            gloo_pg = dist.new_group(backend="gloo")

    torch.multiprocessing.set_sharing_strategy("file_system")

    if dist.get_rank() == src_rank:
        assert tensor is not None, f"src_rank ({src_rank}) must provide a tensor"

        # Intialize a custom pickler
        buf = io.BytesIO()
        shm_pickler = pickle.Pickler(buf)
        shm_pickler.dispatch_table = copyreg.dispatch_table.copy()

        # Register reducers for moving the tensor storage to shared memory
        for t in torch._storage_classes:
            if t.__name__ == "_UntypedStorage":
                # pyre-ignore [16]
                shm_pickler.dispatch_table[t] = reduce_storage
            else:
                shm_pickler.dispatch_table[t] = reduce_typed_storage_child
        # pyre-fixme[16]: Module `storage` has no attribute `_TypedStorage`.
        shm_pickler.dispatch_table[torch.storage._TypedStorage] = reduce_typed_storage

        tensor.share_memory_()
        shm_pickler.dump(tensor)

        obj_list = [buf.getvalue()]
        dist.broadcast_object_list(obj_list, src=src_rank, group=gloo_pg)
        dist.barrier(group=gloo_pg)
        return tensor
    else:
        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=src_rank, group=gloo_pg)
        obj = obj_list[0]
        assert obj is not None
        buf = io.BytesIO(obj)
        dist.barrier(group=gloo_pg)
        return pickle.load(buf)


def main() -> None:
    """
    An example for initializing a torchrec sharded embedding bag with a
    pretrained embedding weight.
    Environment assumptions:
    - The embedding weight fits in the RAM of a single host, but may OOM if all
      processes on the host load the embedding weight simultaneously.
    - For simplicity, the demo assumes a single-host, multi-process environment.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    pg = dist.group.WORLD
    assert pg is not None
    dist.barrier()

    if dist.get_rank() == 0:
        # Pretend that we are loading the pretrained embedding weight from a parquet file on rank 0.
        emb = torch.rand(2000000, 64)
        # Share the tensor to local peers via shared memory
        emb = share_tensor_via_shm(tensor=emb)
    else:
        # Received the tensor shared by rank 0 via shared memory
        emb = share_tensor_via_shm(tensor=None)

    assert emb.is_shared()

    # For demo purpose, the entire model is an embedding bag collection with a
    # single embedding bag.
    ebc = EmbeddingBagCollection(
        device=torch.device("meta"),
        tables=[
            torchrec.EmbeddingBagConfig(
                name="emb",
                embedding_dim=64,
                num_embeddings=2000000,
                feature_names=["f"],
                pooling=torchrec.PoolingType.SUM,
            )
        ],
    )

    # Create a rowwise sharding plan
    sharders = cast(
        List[ModuleSharder[nn.Module]],
        [
            EmbeddingBagCollectionSharder(
                fused_params={
                    "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                    "learning_rate": 0.01,
                    "eps": 0.01,
                }
            )
        ],
    )
    plan = EmbeddingShardingPlanner(
        topology=Topology(world_size=dist.get_world_size(), compute_device=device.type),
        constraints={
            "emb": ParameterConstraints(sharding_types=[ShardingType.ROW_WISE.value])
        },
    ).collective_plan(
        ebc,
        sharders,
        pg,
    )
    print(plan)

    # Initialize dmp which shards the embedding bag
    dmp = DistributedModelParallel(
        module=ebc,
        device=device,
        plan=plan,
        sharders=sharders,
    )
    print(
        "Finished initializing DistributedModelParallel. "
        f"Current device utilization: {torch.cuda.memory_allocated() / 1_000_000} MB"
    )

    # For each shard in sharded tensors, load from the corresponding slice from
    # the pretrained weights in shared memory.
    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            for _, t in dmp.state_dict().items():
                for shard in t.local_shards():
                    offsets = shard.metadata.shard_offsets
                    lengths = shard.metadata.shard_sizes
                    src = emb[
                        offsets[0] : offsets[0] + lengths[0],
                        offsets[1] : offsets[1] + lengths[1],
                    ]
                    shard.tensor.copy_(src)
            dist.barrier()
        else:
            dist.barrier()


def invoke_main() -> None:
    lc = pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=8,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )
    pet.elastic_launch(lc, entrypoint=main)()


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
