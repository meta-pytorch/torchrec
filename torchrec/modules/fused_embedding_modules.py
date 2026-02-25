#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterator, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import torchrec.optim as trec_optim
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    EmbeddingLocation,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    pooling_type_to_pooling_mode,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
    EmbeddingCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.optim.fused import FusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class EmbeddingFusedOptimizer(FusedOptimizer):
    """
    EmbeddingFusedOptimizer exposes the internal SplitTableBatchedEmbeddingBagsCodeGen optimizer state.
    It can be used like a normal optimizer to perform functionalities such as loading/saving optimizers
    and updating learning rates.

    Args:
        tables (List[BaseEmbeddingConfig]): list of embedding tables.
        emb_module (SplitTableBatchedEmbeddingBagsCodeGen): Fbgemm module whose optimizer state we want to expose
    Example:
        See usage in _BatchedFusedEmbeddingLookups
    """

    def __init__(  # noqa C901
        self,
        tables: List[BaseEmbeddingConfig],
        emb_module: SplitTableBatchedEmbeddingBagsCodegen,
    ) -> None:
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = emb_module

        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.get_learning_rate(),
        }

        params: Dict[str, torch.Tensor] = {}

        # Fused optimizers use buffers (they don't use autograd) and we want to make sure
        # that state_dict look identical to no-fused version.
        split_embedding_weights = emb_module.split_embedding_weights()
        split_optimizer_states = emb_module.split_optimizer_states()

        for embedding_weight, optimizer_states, table in zip(
            split_embedding_weights, split_optimizer_states, tables
        ):
            weight = embedding_weight

            state[weight] = {}
            param_group["params"].append(weight)
            param_key = table.name + ".weight"
            params[param_key] = weight

            if len(optimizer_states) >= 1:
                state[weight][f"{param_key}.momentum_1"] = optimizer_states[0]
            if len(optimizer_states) >= 2:
                state[weight][f"{param_key}.momentum_2"] = optimizer_states[1]

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyrefly: ignore[bad-index]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    def step(self, closure: Any = None) -> None:
        # pyrefly: ignore[bad-index]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])


# lint-ignore
class _BatchedFusedEmbeddingLookups(nn.Module, FusedOptimizerModule):
    """
    _BatchedFusedEmbeddingLookups is a thin wrapper we have around SplitTableBatchedEmbeddingBagsCodegen.
    This is not meant to be directly used. Instead use FusedEmbeddingBagCollection/FusedEmbeddingCollection (which in turn utilizes this).

    Example
    -------
    >>> See usage in FusedEmbeddingBagCollection and FusedEmbeddingCollection

    """

    def __init__(
        self,
        embedding_tables: List[BaseEmbeddingConfig],
        data_type: DataType,
        pooling: PoolingType,
        optimizer_type: EmbOptimType,
        optimizer_kwargs: Dict[str, Any],
        device: torch.device,
        embedding_location: EmbeddingLocation,
    ) -> None:
        super().__init__()

        self._rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._cols: List[int] = []
        self._feature_table_map: List[int] = []
        self._emb_names: List[str] = []
        self._embedding_tables = embedding_tables

        for idx, table in enumerate(embedding_tables):
            self._rows.append(table.num_embeddings)
            self._weight_init_mins.append(table.get_weight_init_min())
            self._weight_init_maxs.append(table.get_weight_init_max())
            self._num_embeddings.append(table.num_embeddings)
            self._cols.append(table.embedding_dim)
            self._feature_table_map.extend([idx] * table.num_features())

        compute_device = ComputeDevice.CPU
        if device.type == "cuda":
            compute_device = ComputeDevice.CUDA

        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                list(
                    zip(
                        self._num_embeddings,
                        self._cols,
                        [embedding_location] * len(embedding_tables),
                        [compute_device] * len(embedding_tables),
                    )
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=pooling_type_to_pooling_mode(pooling),
                device=device,
                optimizer=optimizer_type,
                **optimizer_kwargs,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            embedding_tables,
            self._emb_module,
        )

        self._init_parameters()

    def forward(
        self,
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            values (torch.Tensor): Tensor containing bags of indicies into the embedding matrix.
            offsets (torch.Tensor): Starting index position of each sequence
            weights (Optional[torch.Tensor]): weights to use to calculate weighted pooling
        Returns:
            Tensor output of `(B , table_1_embedding_dim + table_2_embedding_dim + ...)`
        """

        return self._emb_module(
            indices=values,
            offsets=offsets,
            per_sample_weights=weights,
        )

    def _init_parameters(self) -> None:
        assert len(self._num_embeddings) == len(
            self._emb_module.split_embedding_weights()
        )
        for rows, emb_dim, weight_init_min, weight_init_max, param in zip(
            self._rows,
            self._cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self._emb_module.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)
            param.data.uniform_(
                weight_init_min,
                weight_init_max,
            )

    def split_embedding_weights(self) -> List[torch.Tensor]:
        return self._emb_module.split_embedding_weights()

    # pyrefly: ignore[bad-override]
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[nn.Parameter]:
        yield cast(nn.Parameter, self._emb_module.weights)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in _BatchedFusedEmbeddingLookups.named_parameters"
        for table, weight in zip(
            self._embedding_tables, self.split_embedding_weights()
        ):
            name = table.name
            key = f"{prefix}.{name}" if (prefix and name) else (prefix + name)
            yield key, cast(nn.Parameter, weight)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in _BatchedFusedEmbeddingLookups.named_buffers"
        for table, param in zip(self._embedding_tables, self.split_embedding_weights()):
            name = f"{table.name}.weight"
            key = f"{prefix}.{name}" if (prefix and name) else (prefix + name)
            yield key, param

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[torch.Tensor]:
        yield from self.split_embedding_weights()

    def flush(self) -> None:
        self._emb_module.flush()


def convert_optimizer_type_and_kwargs(
    optimizer_type: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
) -> Optional[Tuple[EmbOptimType, Dict[str, Any]]]:
    optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
    if "lr" in optimizer_kwargs:
        optimizer_kwargs["learning_rate"] = optimizer_kwargs["lr"]
        optimizer_kwargs.pop("lr")

    if optimizer_type == torch.optim.SGD:
        return (
            EmbOptimType.EXACT_SGD,
            optimizer_kwargs,
        )
    elif optimizer_type == torch.optim.Adagrad:
        return (EmbOptimType.EXACT_ADAGRAD, optimizer_kwargs)
    elif optimizer_type == trec_optim.RowWiseAdagrad:
        return (EmbOptimType.EXACT_ROWWISE_ADAGRAD, optimizer_kwargs)
    elif optimizer_type == torch.optim.Adam:
        return (EmbOptimType.ADAM, optimizer_kwargs)

    return None


class FusedEmbeddingBagCollection(
    EmbeddingBagCollectionInterface, FusedOptimizerModule
):
    """
    FusedEmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).
    It utilizes a technique called Optimizer fusion (register the optimizer with model). The semantics
    of this is that during the backwards pass, the registered optimizer will be called.

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B x F x D] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        optimizer (Type[torch.optim.Optimizer]): fusion optimizer type
        optimizer_kwargs: Dict[str, Any]: fusion optimizer kwargs
        device (Optional[torch.device]): compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=8, num_embeddings=10, feature_names=["f2"]
        )

        ebc = FusedEmbeddingBagCollection(tables=[table_0, table_1], optimizer_type=torch.optim.SGD, optimizer_kwargs={"lr": .01})

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[ 0.2093,  0.1395,  0.1571,  0.3583,  0.0421,  0.0037, -0.0692,  0.0663,
          0.2166, -0.3150, -0.2771, -0.0301],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0165, -0.1225,  0.2483,  0.0624,
         -0.1168, -0.0509, -0.1309,  0.3059],
        [ 0.0811, -0.1779, -0.1443,  0.1097, -0.4410, -0.4036,  0.4458, -0.2735,
         -0.3080, -0.2102, -0.0564,  0.5583]], grad_fn=<CatBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        [0, 4, 12]
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        location: Optional[EmbeddingLocation] = None,
    ) -> None:
        super().__init__()

        self._optimizer_type = optimizer_type
        self._optimizer_kwargs = optimizer_kwargs
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )

        emb_optim_and_kwargs = convert_optimizer_type_and_kwargs(
            optimizer_type, optimizer_kwargs
        )
        if emb_optim_and_kwargs is None:
            raise ValueError(
                f"Cannot fuse optimizer_type={optimizer_type} with kwargs {optimizer_kwargs}"
            )
        (emb_optim_type, emb_opt_kwargs) = emb_optim_and_kwargs

        if location in [
            EmbeddingLocation.DEVICE,
            EmbeddingLocation.MANAGED,
            EmbeddingLocation.MANAGED_CACHING,
        ]:
            assert device is not None and device.type in [
                "cuda",
                "meta",
            ], f"Using location={location} requires device=cuda or meta"

        if device is None:
            device = torch.device("cpu")

        if location is None:
            if device.type in ["cpu", "meta"]:
                location = EmbeddingLocation.HOST
            elif device.type == "cuda":
                location = EmbeddingLocation.DEVICE
            else:
                raise ValueError("EmbeddingLocation could not be set")

        self._is_weighted = is_weighted
        self._embedding_bag_configs = tables

        # Registering in a List instead of ModuleList because we want don't want them to be auto-registered.
        # Their states will be modified via self.embedding_bags
        self._emb_modules: List[nn.Module] = []

        self._key_to_tables: Dict[
            Tuple[PoolingType, DataType], List[EmbeddingBagConfig]
        ] = defaultdict(list)

        self._length_per_key: List[int] = []

        for table in tables:
            self._length_per_key.extend(
                [table.embedding_dim] * len(table.feature_names)
            )

            key = (table.pooling, table.data_type)
            self._key_to_tables[key].append(table)

        optims = []
        for key, tables in self._key_to_tables.items():
            (pooling, data_type) = key
            emb_module = _BatchedFusedEmbeddingLookups(
                cast(List[BaseEmbeddingConfig], tables),
                data_type=data_type,
                pooling=pooling,
                optimizer_type=emb_optim_type,
                optimizer_kwargs=emb_opt_kwargs,
                device=device,
                embedding_location=location,
            )
            self._emb_modules.append(emb_module)
            params: Dict[str, torch.Tensor] = {}
            for param_key, weight in emb_module.fused_optimizer().params.items():
                params[f"embedding_bags.{param_key}"] = weight
            optims.append(("", emb_module.fused_optimizer()))

        # pyrefly: ignore[bad-argument-type]
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._embedding_names: List[str] = list(
            itertools.chain(*get_embedding_names_by_table(self._embedding_bag_configs))
        )

        # We map over the parameters from FBGEMM backed kernels to the canonical nn.EmbeddingBag
        # representation. This provides consistency between this class and the EmbeddingBagCollection's
        # nn.Module API calls (state_dict, named_modules, etc)
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for (_key, tables), emb_module in zip(
            self._key_to_tables.items(), self._emb_modules
        ):
            for embedding_config, weight in zip(
                tables,
                # pyrefly: ignore[not-callable]
                emb_module.split_embedding_weights(),
                #  torch._tensor.Tensor]` is not a function.
            ):
                self.embedding_bags[embedding_config.name] = torch.nn.Module()
                self.embedding_bags[embedding_config.name].register_parameter(
                    "weight", torch.nn.Parameter(weight)
                )

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """
        assert features is not None
        feature_dict = features.to_dict()
        embeddings = []

        for emb_op, (_key, tables) in zip(
            self._emb_modules, self._key_to_tables.items()
        ):
            indicies = []
            lengths = []
            offsets = []
            weights = []

            for table in tables:
                for feature in table.feature_names:
                    f = feature_dict[feature]
                    indicies.append(f.values())
                    lengths.append(f.lengths())
                    if self._is_weighted:
                        weights.append(f.weights())

            indicies = torch.cat(indicies)
            lengths = torch.cat(lengths)

            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            if self._is_weighted:
                weights = torch.cat(weights)

            embeddings.append(
                emb_op(
                    indicies.int(),
                    offsets.int(),
                    weights if self._is_weighted else None,
                )
            )

        embeddings = torch.cat(embeddings, dim=1)

        return KeyedTensor(
            keys=self._embedding_names,
            values=embeddings,
            length_per_key=self._length_per_key,
        )

    def _get_name(self) -> str:
        return "FusedEmbeddingBagCollection"

    @property
    def device(self) -> torch.device:
        return self._device

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def is_weighted(self) -> bool:
        return self._is_weighted

    def optimizer_type(self) -> Type[torch.optim.Optimizer]:
        return self._optimizer_type

    def optimizer_kwargs(self) -> Dict[str, Any]:
        return self._optimizer_kwargs

    # pyrefly: ignore[bad-override]
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim


class FusedEmbeddingCollection(EmbeddingCollectionInterface, FusedOptimizerModule):
    """
    EmbeddingCollection represents a unsharded collection of non-pooled embeddings. The semantics
    of this module is that during the backwards pass, the registered optimizer will be called.

    It processes sparse data in the form of `KeyedJaggedTensor` of the form [F X B X L]
    where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (variable)

    and outputs `Dict[feature (key), JaggedTensor]`.
    Each `JaggedTensor` contains values of the form (B * L) X D
    where:

    * B: batch size
    * L: length of sparse features (jagged)
    * D: each feature's (key's) embedding dimension and lengths are of the form L

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables.
        device (Optional[torch.device]): default compute device.
        need_indices (bool): if we need to pass indices to the final lookup dict.

    Example::

        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        feature_embeddings = ec(features)
        print(feature_embeddings['f2'].values())
        tensor([[-0.2050,  0.5478,  0.6054],
        [ 0.7352,  0.3210, -3.0399],
        [ 0.1279, -0.1756, -0.4130],
        [ 0.7519, -0.4341, -0.0499],
        [ 0.9329, -1.0697, -0.8095]], grad_fn=<EmbeddingBackward>)
    """

    # noqa lint
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
        location: Optional[EmbeddingLocation] = None,
    ) -> None:
        super().__init__()

        self._optimizer_type = optimizer_type
        self._optimizer_kwargs = optimizer_kwargs

        emb_optim_and_kwargs = convert_optimizer_type_and_kwargs(
            optimizer_type, optimizer_kwargs
        )
        if emb_optim_and_kwargs is None:
            raise ValueError(
                f"Cannot fuse optimizer_type={optimizer_type} with kwargs {optimizer_kwargs}"
            )
        (emb_optim_type, emb_opt_kwargs) = emb_optim_and_kwargs

        if location in [
            EmbeddingLocation.DEVICE,
            EmbeddingLocation.MANAGED,
            EmbeddingLocation.MANAGED_CACHING,
        ]:
            assert device is not None and device.type in [
                "cuda",
                "meta",
            ], f"Using location={location} requires device=cuda or meta"

        if device is None:
            device = torch.device("cpu")

        assert device.type in [
            "cuda",
            "meta",
        ], "FusedEmbeddingCollection is only supported for device in [CUDA, meta] currently. There are plans to support device=CPU."

        if location is None:
            if device.type in ["cpu", "meta"]:
                location = EmbeddingLocation.HOST
            elif device.type == "cuda":
                location = EmbeddingLocation.DEVICE
            else:
                raise ValueError("EmbeddingLocation could not be set")

        self._embedding_configs = tables
        self._need_indices: bool = need_indices
        self._embedding_dim: int = -1

        # Registering in a List instead of ModuleList because we want don't want them to be auto-registered.
        # Their states will be modified via self.embedding_bags
        self._emb_modules: List[nn.Module] = []

        self._key_to_tables: Dict[DataType, List[EmbeddingConfig]] = defaultdict(list)

        seen_features = set()
        self._shared_features: Set[str] = set()
        for table in tables:
            key = table.data_type
            self._key_to_tables[key].append(table)

            if self._embedding_dim == -1:
                self._embedding_dim = table.embedding_dim
            elif self._embedding_dim != table.embedding_dim:
                raise ValueError(
                    "All tables in a EmbeddingCollection are required to have same embedding dimension."
                    + f" Violating case: {table}'s embedding_dim {table.embedding_dim} !="
                    + f" {self._embedding_dim}"
                )
            for feature in table.feature_names:
                if feature in seen_features:
                    self._shared_features.add(feature)
                else:
                    seen_features.add(feature)

        optims = []
        for key, tables in self._key_to_tables.items():
            data_type = key
            emb_module = _BatchedFusedEmbeddingLookups(
                cast(List[BaseEmbeddingConfig], tables),
                data_type=data_type,
                pooling=PoolingType.NONE,
                optimizer_type=emb_optim_type,
                optimizer_kwargs=emb_opt_kwargs,
                device=device,
                embedding_location=location,
            )
            self._emb_modules.append(emb_module)
            params: Dict[str, torch.Tensor] = {}
            for param_key, weight in emb_module.fused_optimizer().params.items():
                params[f"embeddings.{param_key}"] = weight
            optims.append(("", emb_module.fused_optimizer()))

        # pyrefly: ignore[bad-argument-type]
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._embedding_names: List[str] = list(
            itertools.chain(*get_embedding_names_by_table(self._embedding_configs))
        )

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            self._embedding_configs,
        )

        # We map over the parameters from FBGEMM backed kernels to the canonical nn.EmbeddingBag
        # representation. This provides consistency between this class and the EmbeddingBagCollection's
        # nn.Module API calls (state_dict, named_modules, etc)
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        for (_key, tables), emb_module in zip(
            self._key_to_tables.items(), self._emb_modules
        ):
            for embedding_config, weight in zip(
                tables,
                # pyrefly: ignore[not-callable]
                emb_module.split_embedding_weights(),
                #  torch._tensor.Tensor]` is not a function.
            ):
                self.embeddings[embedding_config.name] = torch.nn.Module()
                self.embeddings[embedding_config.name].register_parameter(
                    "weight", torch.nn.Parameter(weight)
                )

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """
        assert features is not None
        feature_dict = features.to_dict()

        feature_embeddings: Dict[str, JaggedTensor] = {}

        for emb_op, (_key, tables) in zip(
            self._emb_modules, self._key_to_tables.items()
        ):
            indicies = []
            lengths = []
            offsets = []

            feature_names = []
            feature_lengths = []
            feature_values = []
            splits = []

            for table in tables:
                for feature in table.feature_names:
                    f = feature_dict[feature]
                    indicies.append(f.values())
                    lengths.append(f.lengths())

                    if feature in self._shared_features:
                        feature = f"{feature}@{table.name}"

                    feature_names.append(feature)
                    feature_values.append(f.values())
                    feature_lengths.append(f.lengths())
                    splits.append(torch.sum(feature_lengths[-1]))

            indicies = torch.cat(indicies)
            lengths = torch.cat(lengths)
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

            lookups = emb_op(indicies.int(), offsets.int(), weights=None)
            # pyrefly: ignore[bad-argument-type]
            lookups = torch.split(lookups, split_size_or_sections=splits)

            for feature, lookup, feature_length, values in zip(
                feature_names, lookups, feature_lengths, feature_values
            ):
                feature_embeddings[feature] = JaggedTensor(
                    values=lookup,
                    lengths=feature_length,
                    # hack to return kJT positional indicies in return type.
                    weights=values if self.need_indices() else None,
                )

        return feature_embeddings

    def _get_name(self) -> str:
        return "FusedEmbeddingCollection"

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_configs

    def embedding_names_by_table(self) -> List[List[str]]:
        return self._embedding_names_by_table

    def embedding_dim(self) -> int:
        return self._embedding_dim

    def optimizer_type(self) -> Type[torch.optim.Optimizer]:
        return self._optimizer_type

    def optimizer_kwargs(self) -> Dict[str, Any]:
        return self._optimizer_kwargs

    # pyrefly: ignore[bad-override]
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    def need_indices(self) -> bool:
        return self._need_indices


def fuse_embedding_optimizer(
    model: nn.Module,
    optimizer_type: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
    location: Optional[EmbeddingLocation] = None,
) -> nn.Module:
    """
    Recursively replaces EmbeddingBagCollection and EmbeddingCollection with
    FusedEmbeddingBagCollection and FusedEmbeddingCollection in a model subtree.
    The fused modules will be initialized using the passed in optimizer parameters, and model location.

    Args:
        model: (nn.Module):
        optimizer_type: (Type[torch.optim.Optimizer]):
        optimizer_kwargs: (Dict[Str, Any]):
        device (Optional[torch.device]):
        location: (Optional[EmbeddingLocation]): GPU location placement
    Returns
        nn.Module: input nn.Module with Fused Embedding Modules

    Example::
        ebc = EmbeddingBagCollection()
        my_model = ExampleModel(ebc)
        my_model = fused_embedding_optimizer(my_model, optimizer_type=torch.optim.SGD, optimizer_kwargs={"lr": .01})
        kjt = KeyedJaggedTensor()
        output = my_model(kjt)
    """
    # Replace all EBCs and ECs in a with a corresponding FusedEmbeddingModule.

    # check if top-level module is EBC/EC
    if isinstance(model, EmbeddingBagCollection):
        return FusedEmbeddingBagCollection(
            model.embedding_bag_configs(),
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            location=location,
        )
    if isinstance(model, EmbeddingCollection):
        return FusedEmbeddingCollection(
            model.embedding_configs(),
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            location=location,
        )

    def replace(_model: nn.Module) -> None:
        for child_name, child in _model.named_children():
            if isinstance(child, EmbeddingBagCollection):
                setattr(
                    _model,
                    child_name,
                    FusedEmbeddingBagCollection(
                        tables=child.embedding_bag_configs(),
                        optimizer_type=optimizer_type,
                        optimizer_kwargs=optimizer_kwargs,
                        device=device,
                        location=location,
                    ),
                )
            elif isinstance(child, EmbeddingCollection):
                setattr(
                    _model,
                    child_name,
                    FusedEmbeddingCollection(
                        tables=child.embedding_configs(),
                        optimizer_type=optimizer_type,
                        optimizer_kwargs=optimizer_kwargs,
                        device=device,
                        location=location,
                    ),
                )
            else:
                replace(child)

    replace(model)
    return model
