import queue
import threading
from typing import Dict, List, Union

from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel

from .id_transformer_collection import IDTransformerCollection
from .ps import PSCollection
from .utils import _get_sharded_modules_recursive


__all__ = []


def _create_transformer_thread(transformer: IDTransformerCollection):
    """
    Create a thread for transformer.
    """

    def loop(transformer, input_queue, output_queue):
        while True:
            global_kjt = input_queue.get()
            if global_kjt is None:
                break
            cache_kjt = transformer.transform(global_kjt)
            output_queue.put(cache_kjt)

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    thread = threading.Thread(
        target=loop, args=(transformer, input_queue, output_queue)
    )
    thread.start()
    return thread, input_queue, output_queue


class IDTransformerGroup:
    def __init__(
        self,
        url,
        module: DistributedModelParallel,
        configs_dict: Dict[str, Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]],
        *,
        eviction_config=None,
        transform_config=None,
        ps_config=None,
        parallel=True,
    ):
        """
        IDTransformerGroup stores the IDTransformer for all sharded modules in a DMP module.

        Args:
            url: configuration for PS, e.g. redis://127.0.0.1:6379/?prefix=model.
            module: DMP module that need dynamic embedding.
            configs_dict: a dictionary that maps the module path of the sharded module to its embedding
                configs or embeddingbag configs. The plan of `module` should contain the module path
                in `configs_dict`.
            eviction_config: configuration for eviction policy. Default is `{"type": "mixed_lru_lfu", "ratio": 0.5}`
            transform_config: configuration for the transformer. Default is `{"type": "naive"}`
            ps_config: configuration for the ps. Default is `{"chunk_size": 8 * 1024 * 1024}
            parallel: Whether the IDTransformerCollections will run parallel. When set to True,
                IDTransformerGroup will start a thread for each IDTransformerCollection.

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
            transformers = IDTransformerGroup(
                "redis://127.0.0.1:6379/?prefix=model",
                m,
                { "emb1": config1, "emb2": config2 })

            for label, kjt1, kjt2 in dataset:
                kjts = transformers.transform({ "emb1": kjt1, "emb2": kjt2 })
                kjt1, kjt2 = kjts["emb1"], kjts["emb2"]
                output = m(kjt1, kjt2)
                ...
        """
        self._parallel = parallel

        # get all sharded_modules from plan
        plan = module.plan
        sharded_modules = _get_sharded_modules_recursive(module.module, "", plan)

        self._id_transformer_collections: Dict[str, IDTransformerCollection] = {}
        for path, configs in configs_dict.items():
            if path not in sharded_modules:
                raise ValueError(
                    f"`{path}` in configs dooes not match any sharded modules. "
                    f"Paths for current sharded modules are: {list(sharded_modules.keys())}."
                )
            sharded_module, params_plan = sharded_modules[path]
            ps_collection = PSCollection.fromModule(
                path, sharded_module, params_plan, url, ps_config
            )
            id_transformer_collection = IDTransformerCollection(
                configs, eviction_config, transform_config, ps_collection
            )
            self._id_transformer_collections[path] = id_transformer_collection

        if self._parallel:
            self._threads = {}
            self._input_queues = {}
            self._output_queues = {}
            for path, transformer in self._id_transformer_collections.items():
                thread, input_queue, output_queue = _create_transformer_thread(
                    transformer
                )
                self._threads[path] = thread
                self._input_queues[path] = input_queue
                self._output_queues[path] = output_queue

    def transform(self, kjt_dict: Dict[str, KeyedJaggedTensor]):
        """
        Transform global `KeyedJaggedTensor`s to local ones.

        Args:
            kjt_dict: dict keyed by module path of global kjts.
        Return:
            Dict[str, KeyedJaggedTensor]
            List[torch.classes.tde.FetchHandle]: list of fetch handles to wait.
        """
        result = {}
        fetch_handles = []
        if self._parallel:
            for path, kjt in kjt_dict.items():
                if path not in self._id_transformer_collections:
                    raise ValueError(
                        f"kjt_dict contain invalid path {path}. "
                        f"should be one of {self._id_transformer_collections.keys()}"
                    )
                self._input_queues[path].put(kjt)

            for path in kjt_dict:
                kjt, handles = self._output_queues[path].get()
                result[path] = kjt
                fetch_handles.extend(handles)
        else:
            for path, kjt in kjt_dict.items():
                if path not in self._id_transformer_collections:
                    raise ValueError(
                        f"kjt_dict contain invalid path {path}. "
                        f"should be one of {self._id_transformer_collections.keys()}"
                    )
                kjt, handles = self._id_transformer_collections[path].transform(kjt)
                result[path] = kjt
                fetch_handles.extend(handles)
        return result, fetch_handles

    def save(self):
        for _, id_transformer_collection in self._id_transformer_collections.items():
            id_transformer_collection.save()

    def __contains__(self, path):
        """
        Check if there is transformer for the path.
        """
        return path in self._id_transformer_collections

    def __del__(self):
        """
        Stop the parallel threads
        """
        if self._parallel:
            # stop the threads
            for _, input_queue in self._input_queues.items():
                input_queue.put(None)
