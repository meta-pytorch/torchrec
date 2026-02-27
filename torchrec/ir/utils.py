#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import logging
import operator
from collections import defaultdict
from typing import cast, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.export import Dim, ShapesCollection
from torch.export.dynamic_shapes import _Dim as DIM
from torch.export.unflatten import InterpreterModule
from torch.fx import Node
from torchrec.ir.types import SerializerInterface
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.regroup import KTRegroupAsDict
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# TODO: Replace the default interface with the python dataclass interface
DEFAULT_SERIALIZER_CLS = SerializerInterface
DYNAMIC_DIMS: Dict[str, int] = defaultdict(int)
logger: logging.Logger = logging.getLogger(__name__)


def qualname(m: Union[nn.Module, type[nn.Module]]) -> str:
    if isinstance(m, nn.Module):
        return type(m).__module__ + "." + type(m).__qualname__
    else:
        return m.__module__ + "." + m.__qualname__


def get_device(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.device]:
    """
    Returns the device of the first non-None tensor in the list.
    """
    for t in tensors:
        if t is not None:
            return t.device
    return None


@torch.library.custom_op("torchrec::ir_emb_lookup", mutates_args={})
def ir_emb_lookup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"torch.ops.torchrec.ir_emb_lookup -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("torchrec::ir_emb_lookup")
def ir_emb_lookup_fake(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"ir_emb_lookup_fake -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.custom_op("torchrec::ir_kt_regroup", mutates_args={})
def ir_kt_regroup_impl(
    tensors: List[Optional[torch.Tensor]],
    batch_size: int,
    dims: List[int],
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"torch.ops.torchrec.ir_kt_regroup -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device, dtype=dtype) for dim in dims]


@torch.library.register_fake("torchrec::ir_kt_regroup")
def ir_kt_regroup_fake(
    tensors: List[Optional[torch.Tensor]],
    batch_size: int,
    dims: List[int],
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"ir_kt_regroup_fake -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device, dtype=dtype) for dim in dims]


@torch.library.custom_op("torchrec::ir_dynamic_batch_emb_lookup", mutates_args={})
def ir_dynamic_batch_emb_lookup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(
        f"torch.ops.torchrec.ir_dynamic_batch_emb_lookup -> ({batch_size}, {dims}) {device}"
    )
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("torchrec::ir_dynamic_batch_emb_lookup")
def ir_dynamic_batch_emb_lookup_fake(
    tensors: List[Optional[torch.Tensor]], batch_dize: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    batch_size = torch.library.get_ctx().new_dynamic_size()
    logger.info(f"ir_dynamic_batch_emb_lookup_fake -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.custom_op("torchrec::ir_tbe_lookup", mutates_args={})
def ir_tbe_lookup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"torch.ops.torchrec.ir_tbe_lookup -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("torchrec::ir_tbe_lookup")
def ir_tbe_lookup_fake(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    dynamic_batch_size = torch.library.get_ctx().new_dynamic_size()
    logger.info(f"ir_tbe_lookup_fake -> ({dynamic_batch_size}, {dims}) {device}")
    return [torch.empty(dynamic_batch_size, dim, device=device) for dim in dims]


def encapsulate_ir_modules(
    module: nn.Module,
    serializer: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
    fqn: str = "",
) -> Tuple[nn.Module, List[str]]:
    """
    Takes a module and encapsulate its embedding modules and serializes them to the module buffer.
    Returns the modified module and a list of fqns that had the buffer added, which is needed for torch.export
    The encapsulation is done by using meta_forward function provided by the serializer
    to replace the module's original forward function.
    """
    preserve_fqns: List[str] = []  # fqns of the serialized modules
    children: List[str] = []  # fqns of the children that need further serialization
    # handle current module, and find the children which need further serialization
    typename = qualname(module)
    if typename in serializer.module_to_serializer_cls:
        children = serializer.encapsulate_module(module)
        preserve_fqns.append(fqn)
    else:
        # if the module is not of type serializer, then we check all its children
        children = [child for child, _ in module.named_children()]

    # handle child modules recursively
    for child in children:
        submodule = module.get_submodule(child)
        child_fqn = f"{fqn}.{child}" if len(fqn) > 0 else child
        _, fqns = encapsulate_ir_modules(submodule, serializer, child_fqn)
        preserve_fqns.extend(fqns)
    return module, preserve_fqns


def decapsulate_ir_modules(
    module: nn.Module,
    serializer: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
    device: Optional[torch.device] = None,
    finalize_interpreter_modules: bool = False,
    short_circuit_pytree_ebc_regroup: bool = False,
) -> nn.Module:
    """
    Takes a module and decapsulate its embedding modules by retrieving the buffer.
    Returns the module with restored embedding (sub) modules.
    """
    for child_fqn, child in module.named_children():
        # perform deserialization on the children first, so that we can replace the child module with
        # the deserialized module, and then replace it in the parent
        child = decapsulate_ir_modules(
            module=child, serializer=serializer, device=device
        )
        # replace the child module with deserialized one if applicable
        setattr(module, child_fqn, child)

    # only deserialize if the module has ir_metadata buffer, otherwise return as is
    # we use "ir_metadata" as a convention to identify the deserializable module
    if "ir_metadata" in dict(module.named_buffers()):
        module = serializer.decapsulate_module(module, device)

    if short_circuit_pytree_ebc_regroup:
        module = _short_circuit_pytree_ebc_regroup(module)
        assert finalize_interpreter_modules, "need finalize_interpreter_modules=True"

    if finalize_interpreter_modules:
        for mod in module.modules():
            if isinstance(mod, InterpreterModule):
                mod.finalize()

    return module


def _get_dim(name: str, min: Optional[int] = None, max: Optional[int] = None) -> DIM:
    """
    Returns a Dim object with the given name and min/max. If the name is not unique, it will append a suffix to the name.
    """
    dim = f"{name}_{DYNAMIC_DIMS[name]}"
    DYNAMIC_DIMS[name] += 1
    return Dim(dim, min=min, max=max)


def _has_dim(t: Optional[torch.Tensor]) -> bool:
    return t is not None and t.dim() > 0


def mark_dynamic_kjt(
    kjt: KeyedJaggedTensor,
    shapes_collection: Optional[ShapesCollection] = None,
    variable_length: bool = False,
    variable_batch: bool = False,
    vlen: Optional[DIM] = None,
    llen: Optional[DIM] = None,
) -> ShapesCollection:
    """
    Makes the given KJT dynamic. If it's not variable length, it will only have
    one dynamic dimension, which is the length of the values (and weights).
    If it is variable length, then the lengths and offsets will be dynamic.

    If a shapes collection is provided, it will be updated with the new shapes,
    otherwise a new shapes collection will be created. A passed-in shapes_collection is
    useful if you have multiple KJTs or other dynamic shapes that you want to trace.

    If a dynamic dim/name is provided, it will directly use that dim/name. Otherwise,
    it will use the default name "vlen" for values, and "llen", "lofs" if variable length.
    A passed-in dynamic dim is useful if the dynamic dim is already used in other places.

    variable batch size means the batch size is dynamic during different training iterations
    the batch size for all features are the same within one iteration/batch. so it still follows
    the correlation: len(lengths) == len(keys) * batch_size

    in the variable length scenario, the batch size could be different for each feature within
    the iteration/batch, so it doesn't follow the correlation: len(lengths) == len(keys) * batch_size

    Args:
        kjt (KeyedJaggedTensor): The KJT to make dynamic.
        shapes_collection (Optional[ShapesCollection]): The collection to update.
        variable_length (bool): Whether the KJT is variable length len(lengths) != len(keys) * batch_size
        variable_batch (bool): Whether the KJT is variable batch size, len(lengths) == len(keys) * batch_size, it only works when variable_length is False.
        vlen (Optional[DIM]): The dynamic length for the values. If it's None, it will use the default name "vlen".
        llen (Optional[DIM]): The dynamic length for the lengths, it's only used when variable_length is true. If it's None, it will use the default name "llen".
        batch_size (Optional[DIM]): The dynamic length for the batch_size, it's only used when variable_length and mark_batch_size are both true.
    """
    if shapes_collection is None:
        shapes_collection = ShapesCollection()
    # min=2 to ensure compatibility with dynamic shapes (empty KJT is padded to size 2)
    # This also helps avoid constraint violations during torch.export when guards
    # are added based on observed batch sizes
    vlen = _get_dim("vlen", min=2) if vlen is None else vlen

    if _has_dim(kjt._values):
        if kjt._values.numel() == 0:
            # if the values is empty, we need to set the shape to (2,) to make it compatible with dynamic shape
            # a 0-size dynamic shape will cause error in torch.export.
            # logically when the values is empty, the lengths and offsets should all be zero-value tensors.
            # And this makes the actual values irrelavent to the downstream process.
            kjt._values = torch.ones(
                2, device=kjt._values.device, dtype=kjt._values.dtype
            )
        shapes_collection[kjt._values] = (vlen,)
    if _has_dim(kjt._weights):
        shapes_collection[kjt._weights] = (vlen,)
    if variable_length:
        llen = _get_dim("llen") if llen is None else llen
        if _has_dim(kjt._lengths):
            shapes_collection[kjt._lengths] = (llen,)
        if _has_dim(kjt._offsets):
            shapes_collection[kjt._offsets] = (llen + 1,)
    elif variable_batch:
        # variable batch size means the batch size is dynamic during different training iterations
        # the batch size for all features are the same within one iteration/batch
        #
        # this is fundamentally different from variable length, where the batch size is different
        # for each feature within one iteration/batch
        #
        # it's the user's responsibility to make sure that in a variable batch scenario,
        # the argument variable_batch is only used when setting variable_length to False,
        # otherwise it will lead to unexpected behavior with the dynamic shapes in torch.export
        num_keys = len(kjt.keys())
        if num_keys > 0:
            batch_size = _get_dim("batch_size")
            if _has_dim(kjt._lengths):
                shapes_collection[kjt._lengths] = (batch_size * num_keys,)
            if _has_dim(kjt._offsets):
                shapes_collection[kjt._offsets] = (batch_size * num_keys + 1,)
    return shapes_collection


def move_to_copy_nodes_to_device(
    unflattened_module: nn.Module,
    device: torch.device,
) -> nn.Module:
    """
    Moves all the copy nodes to the given device.
    """
    # pyrefly: ignore[missing-attribute]
    for nodes in unflattened_module.graph.nodes:
        if "_to_copy" in nodes.name:
            new_kwargs = {}
            for k, v in nodes.kwargs.items():
                if isinstance(v, torch.device):
                    v = device
                new_kwargs[k] = v
            nodes.kwargs = new_kwargs

    return unflattened_module


def _check_graph_node(mod: nn.Module, fqn: str) -> bool:
    # pyrefly: ignore[missing-attribute]
    for node in mod.graph.nodes:
        if node.op == "call_module" and node.target == fqn:
            return True
    if "." not in fqn:
        return False
    curr, fqn = fqn.split(".", maxsplit=1)
    mod = getattr(mod, curr)
    return _check_graph_node(mod, fqn)


def _short_circuit_pytree_ebc_regroup(module: nn.Module) -> nn.Module:
    """
    Bypass pytree flatten and unflatten function between EBC and KTRegroupAsDict to avoid key-order issue.
    https://fb.workplace.com/groups/1028545332188949/permalink/1042204770823005/
    EBC ==> (out-going) pytree.flatten ==> tensors and specs ==> (in-coming) pytree.unflatten ==> KTRegroupAsDict
    """
    ebc_fqns: List[str] = []
    regroup_fqns: List[str] = []
    for fqn, m in module.named_modules(remove_duplicate=False):
        if isinstance(m, FeatureProcessedEmbeddingBagCollection):
            ebc_fqns.append(fqn)
        elif isinstance(m, EmbeddingBagCollection):
            if len(ebc_fqns) > 0 and fqn.startswith(ebc_fqns[-1]):
                continue
            ebc_fqns.append(fqn)
        elif isinstance(m, KTRegroupAsDict):
            # check if the KTRegroupAsDict is used. Otherwise we can skip pruning graph.
            if _check_graph_node(module, fqn):
                regroup_fqns.append(fqn)
            else:
                logger.warning(
                    "a KTRegroupAsDict module is ignored from the graph, probably it's replaced "
                    "by a customized regroup module in the forward, and likely there's perf impact."
                )

    if len(ebc_fqns) == len(regroup_fqns) == 0:
        # nothing happens if there is no EBC or KTRegroupAsDict (e.g., the PEA case)
        return module
    elif len(regroup_fqns) == 0:
        # model only contains EBCs, KT (from EBC) pytree.flatten has performance impact
        logger.warning(
            "Expect perf impact if KTRegroupAsDict is not used together with EBCs."
        )
        return module
    elif len(ebc_fqns) == 0:
        # model only contains KTRegroupAsDict, KTs are not from EBC, need to be careful
        logger.warning("KTRegroupAsDict is not from EBC, need to be careful.")
        return module
    else:
        return prune_pytree_flatten_unflatten(
            module, in_fqns=regroup_fqns, out_fqns=ebc_fqns
        )


def prune_pytree_flatten_unflatten(
    module: nn.Module, in_fqns: List[str], out_fqns: List[str]
) -> nn.Module:
    """
    Remove pytree flatten and unflatten function between the given in_fqns and out_fqns.
    "preserved module" ==> (out-going) pytree.flatten ==> [tensors and specs]
        [tensors and specs] ==> (in-coming) pytree.unflatten ==> "preserved module"
    """

    def _get_graph_node(mod: nn.Module, fqn: str) -> Tuple[nn.Module, Node, str]:
        # pyrefly: ignore[missing-attribute]
        for node in mod.graph.nodes:
            if node.op == "call_module" and node.target == fqn:
                return mod, node, fqn
        assert "." in fqn, f"can't find {fqn} in the graph of {mod}"
        curr, fqn = fqn.split(".", maxsplit=1)
        mod = getattr(mod, curr)
        return _get_graph_node(mod, fqn)

    # remove tree_unflatten from the in_fqns (in-coming nodes)
    for fqn in in_fqns:
        submodule, node, submod_name = _get_graph_node(module, fqn)

        # kt_regroup node will have either one arg or one kwarg
        use_args = len(node.args) == 1
        use_kwargs = len(node.kwargs) == 1
        assert use_args or use_kwargs

        # Incase the kt_regroup module is partitioned to a submodule, we need
        # to check the parent module for tree_unflatten node.

        if (use_args and cast(Node, node.args[0]).op == "placeholder") or (
            use_kwargs and cast(Node, list(node.kwargs.values())[0]).op == "placeholder"
        ):
            submodule, node, _ = _get_graph_node(
                module, fqn.replace("." + submod_name, "")
            )
            use_args = len(node.args) == 1
            use_kwargs = len(node.kwargs) == 1
            assert use_args or use_kwargs

        getitem_getitem = cast(
            Node, node.args[0] if use_args else list(node.kwargs.values())[0]
        )
        assert (
            getitem_getitem.op == "call_function"
            and getitem_getitem.target == operator.getitem
        )
        tree_unflatten_getitem = cast(Node, getitem_getitem.args[0])
        assert (
            tree_unflatten_getitem.op == "call_function"
            and tree_unflatten_getitem.target == operator.getitem
        )
        tree_unflatten = cast(Node, tree_unflatten_getitem.args[0])
        assert (
            tree_unflatten.op == "call_function"
            # pyrefly: ignore[implicit-import]
            and tree_unflatten.target == torch.utils._pytree.tree_unflatten
        )
        logger.info(f"Removing tree_unflatten from {fqn}")
        input_nodes = tree_unflatten.args[0]
        if use_args:
            node.args = (input_nodes,)
        else:
            node.kwargs = {list(node.kwargs.keys())[0]: input_nodes}
        #  `eliminate_dead_code`.
        # pyrefly: ignore[missing-attribute]
        submodule.graph.eliminate_dead_code()

    # remove tree_flatten_spec from the out_fqns (out-going nodes)
    for fqn in out_fqns:
        submodule, node, _ = _get_graph_node(module, fqn)
        users = list(node.users.keys())
        assert (
            len(users) == 1
            and users[0].op == "call_function"
            # pyrefly: ignore[implicit-import]
            and users[0].target == torch.fx._pytree.tree_flatten_spec
        )
        tree_flatten_users = list(users[0].users.keys())
        assert (
            len(tree_flatten_users) == 1
            and tree_flatten_users[0].op == "call_function"
            and tree_flatten_users[0].target == operator.getitem
        )
        logger.info(f"Removing tree_flatten_spec from {fqn}")
        getitem_node = tree_flatten_users[0]
        getitem_node.replace_all_uses_with(node)
        #  `eliminate_dead_code`.
        # pyrefly: ignore[missing-attribute]
        submodule.graph.eliminate_dead_code()
    return module
