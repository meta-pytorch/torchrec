#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import inspect
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
    if qualname(module) in serializer.module_to_serializer_cls:
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
    if not hasattr(mod, "graph"):
        return False
    # pyrefly: ignore[missing-attribute]
    for node in mod.graph.nodes:
        if node.op == "call_module" and node.target == fqn:
            return True
    if "." not in fqn:
        return False
    curr, fqn = fqn.split(".", maxsplit=1)
    child = getattr(mod, curr, None)
    if child is None:
        return False
    return _check_graph_node(child, fqn)


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


# ---------------------------------------------------------------------------
# Post-decapsulation graph fixup utilities
# ---------------------------------------------------------------------------


class _SimpleTensorRegroup(nn.Module):
    r"""Drop-in replacement for KTRegroupAsDict after IR pytree pruning.

    After ``_short_circuit_pytree_ebc_regroup`` removes pytree ops, the parent
    graph passes a fused plain tensor instead of ``List[KeyedTensor]``.
    :class:`~torchrec.modules.regroup.KTRegroupAsDict` expects KeyedTensors,
    so this module handles plain tensors via :func:`torch.split`.

    Args:
        splits (List[int]): embedding dimensions per group for splitting the
            fused tensor along dim=1.
        keys (List[str]): original keyed tensor group names.
        used_indices (Optional[List[int]]): if set, only return tensors at
            these indices from the split result. Default: ``None``

    Example::

        >>> regroup = _SimpleTensorRegroup(
        ...     splits=[64, 32],
        ...     keys=["user", "item"],
        ... )
        >>> fused = torch.randn(4, 96)
        >>> out = regroup(fused)
        >>> assert isinstance(out, tuple) and len(out) == 2
    """

    def __init__(
        self,
        splits: List[int],
        keys: List[str],
        used_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._splits = splits
        self._keys = keys
        self._used_indices = used_indices

    # pyre-ignore[3]: Return type should match forward signature.
    def forward(
        self,
        input_tensor: torch.Tensor,
        input_tensor2: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        r"""Split fused tensor along dim=1 and return selected outputs.

        Args:
            input_tensor (torch.Tensor): primary fused embedding tensor.
            input_tensor2 (torch.Tensor, optional): optional second tensor
                from a separate EBC shard, concatenated before splitting.
                Default: ``None``

        Returns:
            Union[Tuple[torch.Tensor, ...], torch.Tensor]: split tensors as
                a tuple, or a single tensor for single-output regroups.
        """
        if input_tensor2 is not None:
            input_tensor = torch.cat([input_tensor, input_tensor2], dim=1)
        elif isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        expected_dim = sum(self._splits)
        actual_dim = input_tensor.size(1)
        n_outputs = (
            len(self._used_indices)
            if self._used_indices is not None
            else len(self._splits)
        )
        if actual_dim != expected_dim:
            if n_outputs == 1:
                return input_tensor
            scale = actual_dim / expected_dim
            scaled = [max(1, int(round(s * scale))) for s in self._splits]
            scaled[-1] = actual_dim - sum(scaled[:-1])
            all_tensors = torch.split(input_tensor, scaled, dim=1)
        else:
            all_tensors = torch.split(input_tensor, self._splits, dim=1)
        if self._used_indices is not None:
            result = tuple(all_tensors[i] for i in self._used_indices)
        else:
            result = all_tensors
        if n_outputs == 1:
            return result[0] if isinstance(result, tuple) else result
        return result


def _get_tbe_arg_order(
    mod: InterpreterModule,
) -> Dict[str, int]:
    """Analyze a TBE InterpreterModule graph to map placeholder names to
    forward parameter positions (indices=0, lengths=1, per_sample_weights=2)
    by finding asynchronous_complete_cumsum (lengths) and ir_tbe_lookup
    (indices, psw) nodes.
    """
    lengths_placeholder = None
    for node in mod.graph.nodes:
        if node.op == "call_function" and "asynchronous_complete_cumsum" in str(
            node.target
        ):
            lengths_placeholder = (
                node.args[0].name if isinstance(node.args[0], torch.fx.Node) else None
            )
            break

    indices_placeholder = None
    psw_placeholder = None
    for node in mod.graph.nodes:
        if node.op == "call_function" and "ir_tbe_lookup" in str(node.target):
            tensors_arg = node.args[0]
            if isinstance(tensors_arg, (list, tuple)) and len(tensors_arg) >= 3:
                if isinstance(tensors_arg[0], torch.fx.Node):
                    indices_placeholder = tensors_arg[0].name
                if tensors_arg[2] is not None and isinstance(
                    tensors_arg[2], torch.fx.Node
                ):
                    psw_placeholder = tensors_arg[2].name
            break

    mapping: Dict[str, int] = {}
    if indices_placeholder:
        mapping[indices_placeholder] = 0
    if lengths_placeholder:
        mapping[lengths_placeholder] = 1
    if psw_placeholder:
        mapping[psw_placeholder] = 2
    return mapping


def _resolve_child_module(parent: nn.Module, target: str) -> Optional[nn.Module]:
    """Resolve a child module by dot-separated target path."""
    try:
        child: nn.Module = parent
        for part in target.split("."):
            child = getattr(child, part)
        return child
    except AttributeError:
        return None


def _classify_placeholders_by_dtype(
    nodes: List[Node],
) -> Tuple[List[str], List[str]]:
    """Classify placeholder nodes as integer or floating-point by dtype."""
    int_phs: List[str] = []
    float_phs: List[str] = []
    for n in nodes:
        val = n.meta.get("val", None) if hasattr(n, "meta") else None
        if val is not None and hasattr(val, "dtype") and val.dtype.is_floating_point:
            float_phs.append(n.name)
        else:
            int_phs.append(n.name)
    return int_phs, float_phs


def _assign_unmapped_positions(
    mapping: Dict[str, int],
    int_phs: List[str],
    float_phs: List[str],
    remaining_names: List[str],
) -> None:
    """Assign indices (pos 0) and psw (pos 2) from placeholder dtype lists."""
    if int_phs and 0 not in mapping.values():
        mapping[int_phs[0]] = 0
    if float_phs and 2 not in mapping.values():
        mapping[float_phs[0]] = 2
    if not int_phs and not float_phs:
        for name in remaining_names:
            if 0 not in mapping.values():
                mapping[name] = 0
            elif 2 not in mapping.values():
                mapping[name] = 2


def _infer_tbe_placeholders_from_parent(
    parent_mod: InterpreterModule,
    mapping: Dict[str, int],
) -> None:
    """Use dtype heuristic on parent module placeholders to infer
    indices (pos 0) and per_sample_weights (pos 2)."""
    placeholders = [n for n in parent_mod.graph.nodes if n.op == "placeholder"]
    lengths_name = None
    for ph_name, pos in mapping.items():
        if pos == 1:
            lengths_name = ph_name
            break
    remaining = [n for n in placeholders if n.name != lengths_name]
    int_phs, float_phs = _classify_placeholders_by_dtype(remaining)
    _assign_unmapped_positions(mapping, int_phs, float_phs, [n.name for n in remaining])


def _capture_tbe_arg_mappings(
    module: nn.Module,
) -> Dict[str, Dict[str, int]]:
    """Before decapsulation, capture placeholder-to-position mappings for
    all TBE InterpreterModules (those with ir_metadata containing
    ir_tbe_lookup or asynchronous_complete_cumsum).

    For IntNBitTableBatchedEmbeddingBagsCodegenWithLength parents, uses
    dtype-based heuristic: int placeholders map to indices (pos 0),
    float placeholders map to per_sample_weights (pos 2).
    """
    mappings: Dict[str, Dict[str, int]] = {}
    for fqn, mod in module.named_modules():
        if not isinstance(mod, InterpreterModule):
            continue
        if "ir_metadata" not in dict(mod.named_buffers(recurse=False)):
            continue
        mapping = _get_tbe_arg_order(mod)
        if mapping:
            mappings[fqn] = mapping
            logger.info(f"TBE arg mapping for {fqn}: {mapping}")

    for fqn in list(mappings.keys()):
        parent_mod = _resolve_child_module(module, fqn)
        if parent_mod is None or not isinstance(parent_mod, InterpreterModule):
            continue
        if not hasattr(parent_mod, "tbe_codegen"):
            continue
        _infer_tbe_placeholders_from_parent(parent_mod, mappings[fqn])
        logger.info(f"Final TBE arg mapping for {fqn}: {mappings[fqn]}")

    return mappings


def _find_regroup_node_and_dims(
    mod: InterpreterModule,
) -> Optional[Tuple[Node, List[int]]]:
    """Find ir_kt_regroup call node and its dims in a module graph."""
    for node in mod.graph.nodes:
        if node.op != "call_function":
            continue
        if "ir_kt_regroup" not in str(node.target):
            continue
        if len(node.args) < 3:
            continue
        raw_dims = node.args[2]
        if isinstance(raw_dims, (list, tuple)):
            return node, [int(d) for d in raw_dims]
    return None


def _collect_getitem_indices(
    out_args: Union[List, Tuple],
    source_node: Node,
) -> List[int]:
    """Collect getitem indices from output args that reference source_node."""
    indices: List[int] = []
    for elem in out_args:
        if not isinstance(elem, torch.fx.Node):
            continue
        if (
            elem.op == "call_function"
            and "getitem" in str(elem.target)
            and len(elem.args) >= 2
            and elem.args[0] is source_node
        ):
            idx = elem.args[1]
            if isinstance(idx, int):
                indices.append(idx)
    return indices


def _find_used_output_indices(
    mod: InterpreterModule,
    regroup_node: Node,
    num_dims: int,
) -> Optional[List[int]]:
    """Find which regroup output indices survive in the module's output."""
    output_nodes = [n for n in mod.graph.nodes if n.op == "output"]
    if not output_nodes:
        return None
    out_args = output_nodes[0].args[0]
    if not isinstance(out_args, (list, tuple)):
        return None
    indices = _collect_getitem_indices(out_args, regroup_node)
    if indices and len(indices) < num_dims:
        return indices
    return None


def _capture_kt_regroup_dims(
    module: nn.Module,
) -> Dict[str, Tuple[List[int], Optional[List[int]]]]:
    """Before decapsulation, capture ir_kt_regroup output dims and which
    output indices are actually used, from encapsulated KTRegroup
    InterpreterModule graphs.
    """
    info_by_fqn: Dict[str, Tuple[List[int], Optional[List[int]]]] = {}
    for fqn, mod in module.named_modules():
        if not isinstance(mod, InterpreterModule):
            continue
        if "ir_metadata" not in dict(mod.named_buffers(recurse=False)):
            continue
        result = _find_regroup_node_and_dims(mod)
        if result is None:
            continue
        regroup_node, dims = result
        used_indices = _find_used_output_indices(mod, regroup_node, len(dims))
        info_by_fqn[fqn] = (dims, used_indices)
        logger.info(
            f"Captured KTRegroup dims for {fqn}: {dims}"
            + (f", used_indices={used_indices}" if used_indices is not None else "")
        )

    return info_by_fqn


def _get_child_expected_args(child: nn.Module) -> Optional[int]:
    """Return expected positional arg count, or None if child accepts *args."""
    if isinstance(child, InterpreterModule):
        return sum(1 for n in child.graph.nodes if n.op == "placeholder")
    sig = inspect.signature(child.forward)
    if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
        return None
    return sum(
        1
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def _trim_single_module_args(mod: InterpreterModule) -> bool:
    """Trim call_module args and remove unused placeholders in one module."""
    changed = False
    for node in mod.graph.nodes:
        if node.op != "call_module":
            continue
        child = _resolve_child_module(mod, node.target)
        if child is None:
            continue
        expected = _get_child_expected_args(child)
        if expected is None:
            continue
        actual = len(node.args)
        if actual > expected:
            node.args = tuple(node.args[actual - expected :])
            changed = True
    for node in list(mod.graph.nodes):
        if node.op == "placeholder" and len(node.users) == 0:
            mod.graph.erase_node(node)
            changed = True
    return changed


def trim_call_module_args(module: nn.Module) -> None:
    """Iteratively trim call_module args to match child module expected
    counts and remove unused placeholders until stable.

    After unflatten, InterpreterModule graphs may have extra args (e.g.,
    sym_size_int from TBE batch_size). This trims them and cascades the
    cleanup upward through the module tree.
    """
    for iteration in range(20):
        changed = False
        for _mod_fqn, mod in module.named_modules():
            if not isinstance(mod, InterpreterModule):
                continue
            if _trim_single_module_args(mod):
                changed = True
        if not changed:
            logger.info(
                f"trim_call_module_args converged after {iteration + 1} iteration(s)"
            )
            break


def _build_reordered_tbe_args(
    mod: InterpreterModule,
    node: Node,
    child: nn.Module,
    mapping: Dict[str, int],
) -> None:
    """Reorder call_module args for a TBE node using pre-decapsulation mapping."""
    from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
        IntNBitTableBatchedEmbeddingBagsCodegen,
    )
    from torchrec.distributed.quant_embedding_kernel import (
        IntNBitTableBatchedEmbeddingBagsCodegenWithLength,
    )

    current_args = list(node.args)
    arg_name_to_node: Dict[str, torch.fx.Node] = {}
    for arg in current_args:
        if isinstance(arg, torch.fx.Node):
            arg_name_to_node[arg.name] = arg

    reordered: List[Optional[torch.fx.Node]] = [None, None, None]
    for placeholder_name, param_pos in mapping.items():
        if placeholder_name in arg_name_to_node:
            reordered[param_pos] = arg_name_to_node[placeholder_name]

    if (
        isinstance(child, IntNBitTableBatchedEmbeddingBagsCodegen)
        and not isinstance(child, IntNBitTableBatchedEmbeddingBagsCodegenWithLength)
        and reordered[1] is not None
    ):
        with mod.graph.inserting_before(node):
            offsets_node = mod.graph.call_function(
                torch.ops.fbgemm.asynchronous_complete_cumsum,
                (reordered[1],),
            )
        reordered[1] = offsets_node

    filtered = [a for a in reordered if a is not None]
    if filtered:
        node.args = tuple(filtered)


def _fix_tbe_output_getitems(
    node: Node,
    mod: InterpreterModule,
) -> List[Node]:
    """Fix getitem unpacking of TBE output. Returns nodes to erase."""
    nodes_to_erase: List[Node] = []
    for user in list(node.users):
        if user.op != "call_function":
            continue
        if user.target is not operator.getitem:
            continue
        idx = user.args[1]
        if idx == 0:
            user.replace_all_uses_with(node)
            nodes_to_erase.append(user)
        elif idx == 1:
            with mod.graph.inserting_after(node):
                batch_size_node = mod.graph.call_method("size", (node, 0))
            user.replace_all_uses_with(batch_size_node)
            nodes_to_erase.append(user)
    return nodes_to_erase


def _reorder_tbe_args(
    module: nn.Module,
    tbe_arg_mappings: Dict[str, Dict[str, int]],
) -> None:
    """After decapsulation, reorder call_module args for TBE modules using
    pre-decapsulation arg mappings, insert cumsum for lengths->offsets where
    needed, and fix output getitem unpacking (real TBE returns Tensor, not
    tuple).
    """
    from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
        IntNBitTableBatchedEmbeddingBagsCodegen,
    )
    from torchrec.distributed.quant_embedding_kernel import (
        IntNBitTableBatchedEmbeddingBagsCodegenWithLength,
    )

    tbe_types = (
        IntNBitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegenWithLength,
    )

    for mod_fqn, mod in module.named_modules():
        if not isinstance(mod, InterpreterModule):
            continue
        nodes_to_erase: List[Node] = []
        for node in mod.graph.nodes:
            if node.op != "call_module":
                continue
            child = _resolve_child_module(mod, node.target)
            if child is None or not isinstance(child, tbe_types):
                continue

            child_fqn = f"{mod_fqn}.{node.target}" if mod_fqn else node.target
            mapping = tbe_arg_mappings.get(child_fqn)
            if mapping:
                _build_reordered_tbe_args(mod, node, child, mapping)
            nodes_to_erase.extend(_fix_tbe_output_getitems(node, mod))

        for n in reversed(nodes_to_erase):
            mod.graph.erase_node(n)


def _is_pytree_pruned(module: nn.Module, regroup_fqn: str) -> bool:
    """Check if pytree pruning happened for a KTRegroupAsDict at the given
    FQN by looking for tree_unflatten in the input chain.
    """
    parts = regroup_fqn.rsplit(".", 1)
    if len(parts) == 2:
        parent_fqn, attr_name = parts
        parent = dict(module.named_modules()).get(parent_fqn)
    else:
        parent = module
        attr_name = regroup_fqn
    if parent is None or not hasattr(parent, "graph"):
        return True

    # pyrefly: ignore[missing-attribute]
    for node in parent.graph.nodes:
        if node.op == "call_module" and node.target == attr_name:
            input_node = None
            if len(node.args) >= 1 and isinstance(node.args[0], torch.fx.Node):
                input_node = node.args[0]
            elif len(node.kwargs) >= 1:
                first_kwarg = next(iter(node.kwargs.values()), None)
                if isinstance(first_kwarg, torch.fx.Node):
                    input_node = first_kwarg

            if input_node is not None:
                cur = input_node
                for _ in range(5):
                    if (
                        cur.op == "call_function"
                        and cur.target == torch.utils._pytree.tree_unflatten
                    ):
                        return False
                    if (
                        cur.op == "call_function"
                        and cur.target == operator.getitem
                        and len(cur.args) >= 1
                        and isinstance(cur.args[0], torch.fx.Node)
                    ):
                        cur = cur.args[0]
                    else:
                        break
            return True
    return True


def _replace_kt_regroup_modules(
    module: nn.Module,
    regroup_dims: Dict[str, Tuple[List[int], Optional[List[int]]]],
) -> None:
    """After decapsulation + pytree pruning, replace KTRegroupAsDict modules
    with _SimpleTensorRegroup where pytree ops were removed and the module
    would receive plain tensors instead of List[KeyedTensor].
    """
    if not regroup_dims:
        return

    replacements: List[Tuple[str, nn.Module, _SimpleTensorRegroup]] = []
    for fqn, mod in module.named_modules():
        if not isinstance(mod, KTRegroupAsDict):
            continue
        # pyre-ignore[16]: _is_inited is an internal attribute.
        if mod._is_inited:
            continue
        if not _is_pytree_pruned(module, fqn):
            logger.info(f"KTRegroupAsDict at {fqn}: pytree not pruned, keeping as-is.")
            continue

        matched_entry = regroup_dims.get(fqn)
        if matched_entry is None:
            for pre_fqn, entry in regroup_dims.items():
                if fqn.endswith(pre_fqn) or pre_fqn.endswith(fqn):
                    matched_entry = entry
                    break
        if matched_entry is None:
            logger.warning(f"KTRegroupAsDict at {fqn} has no matching dims, skipping")
            continue
        matched_dims, used_indices = matched_entry
        # pyre-ignore[16]: _keys is an internal attribute.
        replacement = _SimpleTensorRegroup(
            splits=matched_dims,
            keys=mod._keys,
            used_indices=used_indices,
        )
        replacements.append((fqn, mod, replacement))
        logger.info(
            f"Replacing KTRegroupAsDict at {fqn} with _SimpleTensorRegroup "
            f"(splits={matched_dims}, keys={mod._keys}, used_indices={used_indices})"
        )

    modules_dict = dict(module.named_modules())
    for fqn, _old_mod, new_mod in replacements:
        parts = fqn.rsplit(".", 1)
        if len(parts) == 2:
            parent_fqn, attr_name = parts
            parent = modules_dict[parent_fqn]
            setattr(parent, attr_name, new_mod)
        else:
            setattr(module, fqn, new_mod)


def remove_metadata_assertions(module: nn.Module) -> None:
    """Remove _assert_tensor_metadata nodes from InterpreterModule graphs.

    These assertions are inserted during export based on fake tensor
    metadata which may not match real tensor metadata after decapsulation
    (e.g., dtype differences between ir_tbe_lookup fake impl and real TBE).
    """
    for fqn, mod in module.named_modules():
        if not isinstance(mod, InterpreterModule):
            continue
        nodes_to_erase = []
        for node in mod.graph.nodes:
            if (
                node.op == "call_function"
                and "_assert_tensor_metadata" in str(node.target)
                and len(node.users) == 0
            ):
                nodes_to_erase.append(node)
        for n in reversed(nodes_to_erase):
            mod.graph.erase_node(n)
        if nodes_to_erase:
            logger.info(f"Removed {len(nodes_to_erase)} metadata assertions from {fqn}")


def decapsulate_and_fixup_ir_modules(
    module: nn.Module,
    serializer: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Decapsulate IR modules and apply all necessary graph fixups.

    This is the recommended entry point for restoring modules after IR
    serialization. It handles the full pipeline:
    1. Capture pre-decapsulation TBE/KTRegroup info
    2. Trim extra call_module args (needed before pytree pruning)
    3. Decapsulate (replace InterpreterModules with real modules)
    4. Short-circuit pytree EBC->KTRegroup ops
    5. Finalize InterpreterModules
    6. Reorder TBE args and fix output unpacking
    7. Replace KTRegroupAsDict with tensor-splitting modules
    8. Final arg trim for post-decap real modules
    9. Remove stale metadata assertions
    """
    tbe_arg_mappings = _capture_tbe_arg_mappings(module)
    kt_regroup_dims = _capture_kt_regroup_dims(module)

    trim_call_module_args(module)

    module = decapsulate_ir_modules(
        module=module,
        serializer=serializer,
        device=device,
        finalize_interpreter_modules=False,
        short_circuit_pytree_ebc_regroup=False,
    )

    module = _short_circuit_pytree_ebc_regroup(module)

    for mod in module.modules():
        if isinstance(mod, InterpreterModule):
            mod.finalize()

    _reorder_tbe_args(module, tbe_arg_mappings)
    _replace_kt_regroup_modules(module, kt_regroup_dims)
    trim_call_module_args(module)
    remove_metadata_assertions(module)

    # pyre-ignore[29]: `Module` is not a function.
    module.finalize()

    return module
