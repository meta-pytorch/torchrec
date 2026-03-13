#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
from typing import Dict, List, Union

import torch
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torchrec import EmbeddingBagConfig, EmbeddingConfig
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .id_transformer_group import IDTransformerGroup


__all__ = ["wrap", "save"]


# Similar to torch.utils.data._utils.pin_memory._pin_memory_loop
def transform_loop(dataloader, transform_fn, out_queue, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    for data in dataloader:
        if done_event.is_set():
            break
        transformed_data = transform_fn(data)

        while not done_event.is_set():
            try:
                out_queue.put(transformed_data, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        # save memory
        del transformed_data

    if not done_event.is_set():
        done_event.set()


class DataLoaderIter:
    def __init__(self, dataloader, transform_fn, num_prefetch=0):
        self._data_queue = queue.Queue(maxsize=num_prefetch)
        self._done_event = threading.Event()
        self._transform_thread = threading.Thread(
            target=transform_loop,
            args=(dataloader, transform_fn, self._data_queue, self._done_event),
        )
        self._transform_thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_data()

    def __del__(self):
        self._done_event.set()

    def _get_data(self):
        if self._done_event.is_set():
            raise StopIteration
        if not self._transform_thread.is_alive():
            raise RuntimeError("Transform thread exited unexpectedly")
        data, handles = self._data_queue.get()
        for handle in handles:
            handle.wait()
        return data


class DataLoader:
    def __init__(
        self,
        id_transformer_group: IDTransformerGroup,
        dataloader,
        *,
        data_info: Dict[int, str] = None,
        paths: List[str] = None,
        num_prefetch=0,
    ):
        self._id_transformer_group = id_transformer_group

        if data_info is not None:
            for _, path in data_info.items():
                if path not in self._id_transformer_group:
                    raise ValueError(
                        f"invalid path `{path}` data_info. No id transformer for this path."
                    )
        else:
            self._paths = paths

        self._data_info = data_info

        self._data_queue = queue.Queue(maxsize=num_prefetch)
        self._done_event = threading.Event()

        self._dataloader = dataloader
        self._num_prefetch = num_prefetch

    def _transform_fn(self, data):
        """
        transform data with `data_info`
        """
        if self._data_info is None:
            data_info = {}
            path_idx = 0
            for i in range(len(data)):
                if isinstance(data[i], KeyedJaggedTensor):
                    if path_idx >= len(self._paths):
                        raise ValueError(
                            "Has more KJT in a data sample than the number of modules, "
                            "could not infer data_info, please set data_info manually"
                        )
                    data_info[i] = self._paths[path_idx]
                    path_idx += 1
        else:
            data_info = self._data_info
        global_kjts = {path: data[idx] for idx, path in data_info.items()}
        cache_kjts, fetch_handles = self._id_transformer_group.transform(global_kjts)
        data = list(data)
        for idx, path in data_info.items():
            data[idx] = cache_kjts[path]
        return tuple(data), fetch_handles

    def __iter__(self):
        return DataLoaderIter(
            self._dataloader, self._transform_fn, num_prefetch=self._num_prefetch
        )

    def __len__(self):
        return len(self._dataloader)


def wrap(
    url: str,
    dataloader,
    module: DistributedModelParallel,
    configs_dict: Dict[str, Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]],
    *,
    data_info: Dict[int, str] = None,
    eviction_config=None,
    transform_config=None,
    ps_config=None,
    parallel=True,
    num_prefetch=0,
):
    """
    DataLoader to transform data from global id to cache id.

    Args:
        url: configuration for PS, e.g. redis://127.0.0.1:6379/?prefix=model.
        dataloader: dataloader to transform.
        module: DMP module that need dynamic embedding.
        configs_dict: a dictionary that maps the module path of the sharded module to its embedding
            configs or embeddingbag configs. The plan of `module` should contain the module path
            in `configs_dict`.
        data_info: dict keyed by int index of module path. For example, if the dataloader produces
            `label, kjt1, kjt2` each iteration and `kjt1` and `kjt2` are inputs to modules of path
            `emb1` and `emb2` respectively, then `data_info` should be `{ 1: "emb1", 2: "emb2" }`.
        eviction_config: configuration for eviction policy. Default is `{"type": "mixed_lru_lfu", "ratio": 0.5}`
        transform_config: configuration for the transformer. Default is `{"type": "naive"}`
        ps_config: configuration for the ps. Default is `{"chunk_size": 8 * 1024 * 1024}
        parallel: Whether the IDTransformerCollections will run paralell. When set to True,
            IDTransformerGroup will start a thread for each IDTransformerCollection.
        num_prefetch: number of samples to prefetch.

    Return:
        DataLoader: the dataloader to transform data.
        DistributedModelParallel: model with id_transformer_group attached.

    Example:
        class Model(nn.Module):
            def __init__(self, config1, config2):
                super().__init__()
                self.emb1 = EmbeddingCollection(tables=config1, device=torch.device("meta"))
                self.emb2 = EmbeddingCollection(tables=config2, device=torch.device("meta"))
                ...

            def forward(self, kjt1, kjt2):
                ...

        m = Model(config1, config2)
        m = DistributedModelParallel(m)
        dataloader, m = tde.wrap("redis://127.0.0.1:6379/", dataloader, m, { "emb1": config1, "emb2": config2 })

        for label, kjt1, kjt2 in dataloader:
            output = m(kjt1, kjt2)
            ...
    """
    id_transformer_group = IDTransformerGroup(
        url,
        module,
        configs_dict,
        eviction_config=eviction_config,
        transform_config=transform_config,
        ps_config=ps_config,
        parallel=parallel,
    )
    paths = list(configs_dict.keys())
    # Attach the id transformer group to module for saving.
    module._id_transformer_group = id_transformer_group

    return (
        DataLoader(
            id_transformer_group=id_transformer_group,
            dataloader=dataloader,
            data_info=data_info,
            paths=paths,
            num_prefetch=num_prefetch,
        ),
        module,
    )


def save(module: DistributedModelParallel):
    """
    Save the dynamic embedding part of the model.
    """
    if not hasattr(module, "_id_transformer_group"):
        raise ValueError(
            "No _id_transformer_group property for module, is this a module with dynamic embeding?"
        )
    if not isinstance(module._id_transformer_group, IDTransformerGroup):
        raise ValueError(
            "module._id_transformer_group property is not IDTransformerGroup, is this a module with dynamic embeding?"
        )

    module._id_transformer_group.save()
