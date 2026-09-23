#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The Maglev authoring API: layers and the model that chains them.

Everything here is parallelism-free -- a model built from these runs end to end
in a single process. Cutting a model into pipeline stages, binding those stages
to process groups, and sharding them lives in
:mod:`torchrec.distributed.maglev.stage`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import (
    Any,
    ClassVar,
    Generic,
    get_args,
    get_origin,
    get_type_hints,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn

# The carrier between Maglev layers: always a tuple of tensors, empty for a
# layer with no incoming activation (the first layer of a model).
Activations = Tuple[torch.Tensor, ...]

_NO_METADATA = object()

TStructuredActivations = TypeVar(
    "TStructuredActivations", bound="StructuredActivations"
)


@dataclass(frozen=True)
class StructuredActivations:
    """Marker base for values crossing a structured Maglev boundary."""


@dataclass(frozen=True)
class _ValueLayout:
    kind: str
    children: Tuple["_ValueLayout", ...] = ()
    keys: Tuple[Any, ...] = ()
    metadata: Any = None
    field_name: Optional[str] = None


def _is_builtin_layout_type(expected_type: Any) -> bool:
    if expected_type in (torch.Tensor, int, str):
        return True
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is list:
        return len(args) == 1 and _is_builtin_layout_type(args[0])
    if origin is dict:
        return (
            len(args) == 2
            and args[0] in (int, str)
            and _is_builtin_layout_type(args[1])
        )
    if origin is tuple:
        child_types = args[:1] if len(args) == 2 and args[1] is Ellipsis else args
        return bool(child_types) and all(
            _is_builtin_layout_type(child_type) for child_type in child_types
        )
    return False


def _type_requires_metadata(expected_type: Any) -> bool:
    if expected_type is torch.Tensor:
        return False
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
        return any(_type_requires_metadata(child_type) for child_type in args)
    return True


def get_structured_activations_layout_metadata_fields(
    activations_type: type[TStructuredActivations],
) -> Tuple[str, ...]:
    """Return the ordered fields required by a structured layout constructor.

    The result contains each dataclass field whose layout cannot be recovered
    from its annotation alone. Callers must provide a same-named keyword when
    constructing ``StructuredActivationsLayout[activations_type]``. Tensor-only
    fields and fixed tensor tuples are omitted. Lists, dictionaries, static
    ``int`` and ``str`` values, metadata-bearing nested containers, and fields
    with custom pack and unpack methods are included.

    Raises ``TypeError`` when ``activations_type`` is not a
    ``StructuredActivations`` subtype, when a custom field defines only one of
    its pack and unpack methods, or when an unsupported field has no custom
    methods.
    """
    if not isinstance(activations_type, type) or not issubclass(
        activations_type, StructuredActivations
    ):
        raise TypeError("activations type must inherit StructuredActivations")

    type_hints = get_type_hints(activations_type)
    metadata_fields: List[str] = []
    for field in fields(activations_type):
        expected_type = type_hints.get(field.name, field.type)
        has_custom_pack = hasattr(activations_type, f"_pack_{field.name}")
        has_custom_unpack = hasattr(activations_type, f"_unpack_{field.name}")
        if has_custom_pack != has_custom_unpack:
            raise TypeError(f"field {field.name!r} requires both custom methods")
        if has_custom_pack:
            metadata_fields.append(field.name)
        elif not _is_builtin_layout_type(expected_type):
            raise TypeError(f"field {field.name!r} requires custom methods")
        elif _type_requires_metadata(expected_type):
            metadata_fields.append(field.name)
    return tuple(metadata_fields)


def _check_constant(value: Any, expected_type: Any, what: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(f"{what} must be {expected_type.__name__}")


def _build_sequence_layout(
    kind: str,
    child_type: Any,
    metadata: Any,
    what: str,
) -> _ValueLayout:
    if _type_requires_metadata(child_type):
        if not isinstance(metadata, (list, tuple)):
            raise TypeError(f"{what} metadata must describe each element")
        child_metadata = metadata
    else:
        if type(metadata) is not int or metadata < 0:
            raise TypeError(f"{what} metadata must be a non-negative length")
        child_metadata = (_NO_METADATA,) * metadata
    return _ValueLayout(
        kind,
        tuple(
            _build_value_layout(child_type, item_metadata, what)
            for item_metadata in child_metadata
        ),
    )


def _build_tuple_layout(expected_type: Any, metadata: Any, what: str) -> _ValueLayout:
    args = get_args(expected_type)
    if len(args) == 2 and args[1] is Ellipsis:
        return _build_sequence_layout("tuple", args[0], metadata, what)
    if not any(_type_requires_metadata(child_type) for child_type in args):
        child_metadata = (_NO_METADATA,) * len(args)
    else:
        if not isinstance(metadata, (list, tuple)) or len(metadata) != len(args):
            raise TypeError(f"{what} metadata must match its tuple annotation")
        child_metadata = tuple(
            item_metadata if _type_requires_metadata(child_type) else _NO_METADATA
            for child_type, item_metadata in zip(args, metadata)
        )
    return _ValueLayout(
        "tuple",
        tuple(
            _build_value_layout(child_type, item_metadata, what)
            for child_type, item_metadata in zip(args, child_metadata)
        ),
    )


def _build_dict_layout(expected_type: Any, metadata: Any, what: str) -> _ValueLayout:
    key_type, value_type = get_args(expected_type)
    if _type_requires_metadata(value_type):
        if not isinstance(metadata, dict):
            raise TypeError(f"{what} metadata must be an ordered dict")
        items = tuple(metadata.items())
    else:
        if not isinstance(metadata, (list, tuple)):
            raise TypeError(f"{what} metadata must be an ordered key sequence")
        items = tuple((key, _NO_METADATA) for key in metadata)
    for key, _ in items:
        _check_constant(key, key_type, f"{what} key")
    return _ValueLayout(
        "dict",
        tuple(
            _build_value_layout(value_type, child_metadata, what)
            for _, child_metadata in items
        ),
        tuple(key for key, _ in items),
    )


def _build_value_layout(
    expected_type: Any,
    metadata: Any,
    what: str,
) -> _ValueLayout:
    if expected_type is torch.Tensor:
        if metadata is not _NO_METADATA:
            raise TypeError(f"{what} does not accept metadata")
        return _ValueLayout("tensor")
    if expected_type in (int, str):
        _check_constant(metadata, expected_type, f"{what} metadata")
        return _ValueLayout("constant", metadata=metadata)
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is list:
        return _build_sequence_layout("list", args[0], metadata, what)
    if origin is tuple:
        return _build_tuple_layout(expected_type, metadata, what)
    if origin is dict:
        return _build_dict_layout(expected_type, metadata, what)
    raise TypeError(f"{what} requires custom pack and unpack methods")


def _pack_value(layout: _ValueLayout, value: Any, owner: Any) -> Activations:
    if layout.kind == "tensor":
        return (value,)
    if layout.kind == "constant":
        if value != layout.metadata:
            raise ValueError(
                f"expected static value {layout.metadata!r}, got {value!r}"
            )
        return ()
    if layout.kind in ("list", "tuple"):
        expected_container = list if layout.kind == "list" else tuple
        if not isinstance(value, expected_container) or len(value) != len(
            layout.children
        ):
            raise ValueError(f"value does not match {layout.kind} layout")
        return tuple(
            tensor
            for child, item in zip(layout.children, value)
            for tensor in _pack_value(child, item, owner)
        )
    if layout.kind == "dict":
        if not isinstance(value, dict) or set(value) != set(layout.keys):
            raise ValueError("value does not match dict layout")
        return tuple(
            tensor
            for key, child in zip(layout.keys, layout.children)
            for tensor in _pack_value(child, value[key], owner)
        )
    packer = getattr(owner, f"_pack_{layout.field_name}")
    return packer(value, layout.metadata)


def _unpack_value(
    layout: _ValueLayout,
    activations: Activations,
    activations_type: type[Any],
) -> Tuple[Any, Activations]:
    if layout.kind == "tensor":
        if not activations:
            raise ValueError("not enough tensors to unpack structured activations")
        return activations[0], activations[1:]
    if layout.kind == "constant":
        return layout.metadata, activations
    if layout.kind in ("list", "tuple", "dict"):
        values: List[Any] = []
        remaining = activations
        for child in layout.children:
            value, remaining = _unpack_value(child, remaining, activations_type)
            values.append(value)
        if layout.kind == "list":
            return values, remaining
        if layout.kind == "tuple":
            return tuple(values), remaining
        return dict(zip(layout.keys, values)), remaining
    unpacker = getattr(activations_type, f"_unpack_{layout.field_name}")
    return unpacker(activations, layout.metadata)


class StructuredActivationsLayout(Generic[TStructuredActivations]):
    """Static layout that maps a structured value to a tensor-only carrier.

    First define a frozen dataclass that inherits ``StructuredActivations``.
    Specializing this class with that dataclass produces a cached concrete
    layout class. Instantiate the layout with static metadata named after the
    activation fields that need it, then reuse the instance for every batch::

        @dataclass(frozen=True)
        class Boundary(StructuredActivations):
            hidden: torch.Tensor
            experts: List[torch.Tensor]
            named: Dict[str, torch.Tensor]
            dimensions: List[int]
            label: str

        BoundaryLayout = StructuredActivationsLayout[Boundary]
        layout = BoundaryLayout(
            experts=2,
            named=("user", "ad"),
            dimensions=[64, 128],
            label="frontend",
        )

        packed = layout.pack(
            Boundary(
                hidden=hidden,
                experts=[expert_0, expert_1],
                named={"ad": ad, "user": user},
                dimensions=[64, 128],
                label="frontend",
            )
        )
        restored = layout.unpack(packed)

    ``get_structured_activations_layout_metadata_fields(Boundary)`` returns the
    exact constructor fields required by the layout. In this example, those are
    ``("experts", "named", "dimensions", "label")``.

    The same-name mapping and metadata formats are:

    * ``Tensor`` is one carrier tensor and takes no constructor argument.
    * A fixed tuple is described by its type arguments. A tensor-only tuple
      takes no metadata.
    * A homogeneous list or variadic tuple whose element layout needs no
      metadata takes its non-negative length. Otherwise it takes a list of the
      metadata required by each element. For example, ``List[Tensor]`` takes
      ``3``, while ``List[int]`` takes the actual static values ``[2, 4, 8]``.
    * A dict whose values need no metadata takes an ordered key sequence. If its
      values need metadata, it takes an insertion-ordered dict from each key to
      that value's metadata. Thus ``Dict[str, Tensor]`` takes ``("a", "b")``,
      while ``Dict[str, int]`` takes ``{"a": 2, "b": 4}``.
    * ``int`` and ``str`` are static values stored in the layout and restored
      without occupying the tensor carrier.

    These rules compose recursively. Metadata for a fixed tuple containing
    metadata-bearing children mirrors the tuple; positions for tensor-only
    children may be ``None``. Dict keys are restricted to ``int`` and ``str``.

    A field with another type must define both methods below on its activation
    dataclass. Defining them also overrides built-in handling for that field::

        def _pack_payload(
            self,
            value: Payload,
            metadata: PayloadMetadata,
        ) -> Activations: ...

        @classmethod
        def _unpack_payload(
            cls,
            activations: Activations,
            metadata: PayloadMetadata,
        ) -> Tuple[Payload, Activations]: ...

    The layout constructor requires a ``payload=...`` argument and passes it to
    both methods unchanged. Custom unpackers must return the reconstructed value
    and the unconsumed activation suffix.

    Layout metadata must be invariant across batches and identical on both sides
    of a pipeline boundary. It is neither inferred from runtime tensors nor sent
    with each microbatch. Values such as dynamic integers must therefore be
    represented as tensors instead of static layout metadata. Construction
    rejects missing or unknown metadata; packing validates container structure
    and constants; unpacking rejects missing or unconsumed tensors.
    """

    _activations_type: ClassVar[type[Any]]
    _specializations: ClassVar[dict[type[Any], type[Any]]] = {}

    def __class_getitem__(cls, activations_type: Any) -> Any:
        if not isinstance(activations_type, type) or not issubclass(
            activations_type, StructuredActivations
        ):
            raise TypeError("layout type must inherit StructuredActivations")
        specialized = cls._specializations.get(activations_type)
        if specialized is None:
            specialized = type(
                f"{activations_type.__name__}Layout",
                (cls,),
                {
                    "_activations_type": activations_type,
                    "__module__": activations_type.__module__,
                },
            )
            cls._specializations[activations_type] = specialized
        return specialized

    def __init__(self, **metadata: Any) -> None:
        type_hints = get_type_hints(self._activations_type)
        metadata_fields = set(
            get_structured_activations_layout_metadata_fields(self._activations_type)
        )
        remaining_metadata = dict(metadata)
        field_layouts: List[Tuple[str, _ValueLayout]] = []
        for field in fields(self._activations_type):
            expected_type = type_hints.get(field.name, field.type)
            has_custom_pack = hasattr(self._activations_type, f"_pack_{field.name}")
            if field.name in metadata_fields and field.name not in remaining_metadata:
                raise TypeError(f"missing metadata for field {field.name!r}")
            if has_custom_pack or not _is_builtin_layout_type(expected_type):
                layout = _ValueLayout(
                    "custom",
                    metadata=remaining_metadata.pop(field.name),
                    field_name=field.name,
                )
            else:
                field_metadata = remaining_metadata.pop(field.name, _NO_METADATA)
                layout = _build_value_layout(
                    expected_type,
                    field_metadata,
                    f"field {field.name!r}",
                )
            field_layouts.append((field.name, layout))
        if remaining_metadata:
            names = ", ".join(sorted(remaining_metadata))
            raise TypeError(f"unknown structured activation metadata: {names}")
        self._field_layouts = tuple(field_layouts)

    def pack(self, value: TStructuredActivations) -> Activations:
        if not isinstance(value, self._activations_type):
            raise TypeError(f"value must be {self._activations_type.__name__}")
        return tuple(
            tensor
            for field_name, layout in self._field_layouts
            for tensor in _pack_value(layout, getattr(value, field_name), value)
        )

    def unpack(self, activations: Activations) -> TStructuredActivations:
        remaining = activations
        values: dict[str, Any] = {}
        for field_name, layout in self._field_layouts:
            value, remaining = _unpack_value(
                layout,
                remaining,
                self._activations_type,
            )
            values[field_name] = value
        if remaining:
            raise ValueError(f"{len(remaining)} activation tensors were not consumed")
        return self._activations_type(**values)


@dataclass(frozen=True)
class ActivationSpec:
    """Description of one tensor in an inter-layer activation tuple.

    A layer declares the activations it consumes and produces as specs
    (:meth:`MaglevLayer.in_activation_specs` /
    :meth:`MaglevLayer.out_activation_specs`) so the pipeline-parallel connector
    can allocate receive buffers -- and match send/recv order -- without running
    a shape-inference forward pass.

    A fixed spec stores the full tensor shape. A batch-size-dependent spec uses
    ``-1`` for its batch dimension, which is materialized from the microbatch
    being executed.

    Args:
        shape: tensor shape, with at most one ``-1`` batch-size dimension.
        dtype: dtype of the tensor. Default is ``torch.float32``.

    Example::

        ActivationSpec(torch.Size([1024, 512]))
        ActivationSpec(torch.Size([-1, 512]))
    """

    shape: torch.Size
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.shape.count(-1) > 1:
            raise ValueError("an activation spec may have at most one batch dimension")
        if any(dim < -1 for dim in self.shape):
            raise ValueError("activation spec dimensions must be non-negative or -1")

    @property
    def batch_size_dependent(self) -> bool:
        """Whether ``shape`` contains a runtime batch-size dimension."""
        return -1 in self.shape

    def materialize_shape(self, batch_size: Optional[int] = None) -> torch.Size:
        """Return the concrete shape for one microbatch."""
        if not self.batch_size_dependent:
            return self.shape
        if batch_size is None or batch_size < 0:
            raise ValueError(
                "a non-negative batch size is required for a batch-size-dependent "
                "activation"
            )
        return torch.Size(batch_size if dim == -1 else dim for dim in self.shape)

    @property
    def requires_grad(self) -> bool:
        """Whether this activation carries a gradient back to the producer.

        Only floating-point activations are differentiable, so integer carriers
        (ids, lengths, offsets) are skipped by the backward hand-off.
        """
        return self.dtype.is_floating_point


def check_activations(
    specs: Sequence[ActivationSpec], activations: Activations, what: str
) -> None:
    """Validate an activation tuple against the specs that describe it.

    Args:
        specs: the declared specs.
        activations: the tensors to check, index-aligned with ``specs``.
        what: label used in the error message (e.g. ``"layer 2 input"``).

    Raises:
        ValueError: if the count, a shape, or a dtype does not match.
    """
    if len(activations) != len(specs):
        raise ValueError(
            f"{what}: expected {len(specs)} activation tensors, got {len(activations)}"
        )
    for i, (spec, tensor) in enumerate(zip(specs, activations)):
        expected_shape = spec.materialize_shape(
            tensor.shape[0] if spec.batch_size_dependent and tensor.ndim > 0 else None
        )
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{what}[{i}]: expected shape {tuple(expected_shape)}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != spec.dtype:
            raise ValueError(
                f"{what}[{i}]: expected dtype {spec.dtype}, got {tensor.dtype}"
            )


def activation_specs_from_tensors(
    activations: Activations,
    batch_size: Optional[int],
    floating_dtype: Optional[torch.dtype] = None,
) -> Tuple[ActivationSpec, ...]:
    """Create a communication contract from representative tensors.

    Passing ``None`` for ``batch_size`` makes every non-scalar tensor depend on
    the runtime microbatch size and records only its trailing dimensions.
    """
    specs: List[ActivationSpec] = []
    for activation in activations:
        shape = list(activation.shape)
        batch_size_dependent = bool(shape) and batch_size is None
        if batch_size_dependent:
            shape[0] = -1
        elif shape:
            assert batch_size is not None
            shape[0] = batch_size
        dtype = (
            floating_dtype
            if floating_dtype is not None and activation.dtype.is_floating_point
            else activation.dtype
        )
        specs.append(ActivationSpec(torch.Size(shape), dtype))
    return tuple(specs)


def cast_activations(
    activations: Activations,
    dtype: Optional[torch.dtype],
    specs: Sequence[ActivationSpec] = (),
) -> Activations:
    """Cast differentiable activation tensors while preserving integer carriers."""
    if dtype is None:
        return activations
    if specs:
        if len(activations) != len(specs):
            raise ValueError(
                "activation count must match the static communication contract"
            )
        return tuple(
            activation.to(dtype) if spec.dtype.is_floating_point else activation
            for activation, spec in zip(activations, specs)
        )
    return tuple(
        activation.to(dtype) if activation.dtype.is_floating_point else activation
        for activation in activations
    )


class MaglevLayer(nn.Module, abc.ABC):
    r"""MaglevLayer(sparse=None, dense=None)

    Base class for a Maglev layer -- the unit of compute in a Maglev model.

    A layer consumes two things and produces one:

    * ``layer_input`` -- its own slice of the batch (its feature partition), of
      whatever type the layer defines (e.g. a ``ModelInput``);
    * ``in_activations`` -- the previous layer's output, always a **tuple of
      tensors** (empty for the first layer of a model);
    * and returns its own output activation, again a tuple of tensors.

    The activation tuples are declared statically by
    :meth:`in_activation_specs` and :meth:`out_activation_specs`. The
    pipeline-parallel connector reads those specs to size its receive buffers and
    to order the per-tensor send/recv pairs, so a layer's declared specs must
    match what its ``forward`` actually consumes and produces
    (:func:`check_activations` asserts this).

    A *stage* -- the unit of pipeline parallelism, one per hardware scale-up
    domain (HSD) -- is one or more consecutive layers bound to a process group by
    :class:`~torchrec.distributed.maglev.stage.StageWrapper`, which takes the
    layer list directly. There is no separate stage type: a stage is just a run
    of layers plus the wrapper that distributes it.

    A layer may expose two structural halves:

    * ``sparse(layer_input)`` performs work independent of the incoming
      activation, such as embedding pooling. It may be ``None``.
    * ``dense(sparse_output, layer_input, in_activations)`` performs the
      remaining work and returns the layer's activation tuple.

    The default :meth:`forward` composes those halves. Existing subclasses may
    continue to override :meth:`forward` and omit both halves; schedules that
    need the split call :meth:`require_dense` and receive a descriptive error.

    Args:
        sparse (nn.Module, optional): Module that consumes ``layer_input`` and
            produces the sparse output. Default: ``None``
        dense (nn.Module, optional): Module that consumes the sparse output,
            ``layer_input``, and incoming activations. Default: ``None``

    Example::

        class BlockDense(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.lin = nn.Linear(dim, dim)

            def forward(self, sparse_output, layer_input, in_activations=()):
                x = self.lin(layer_input)
                if in_activations:
                    x = x + in_activations[0]
                return (x,)

        class Block(MaglevLayer):
            def __init__(self, dim: int, is_first: bool) -> None:
                super().__init__(dense=BlockDense(dim))
                self._dim = dim
                self._is_first = is_first

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                if self._is_first:
                    return ()
                return (ActivationSpec(torch.Size([-1, self._dim])),)

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, self._dim])),)

    """

    def __init__(
        self,
        sparse: Optional[nn.Module] = None,
        dense: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.sparse: Optional[nn.Module] = sparse
        self.dense: Optional[nn.Module] = dense

    def require_dense(self) -> nn.Module:
        r"""require_dense() -> nn.Module

        Return the dense half required by split-aware schedules.

        Returns:
            nn.Module: The layer's dense half.

        Raises:
            ValueError: If the layer was built without a dense half.
        """
        if self.dense is None:
            raise ValueError(
                f"{type(self).__name__} was built without a dense half; provide "
                "dense= to MaglevLayer or use a schedule that executes whole layers"
            )
        return self.dense

    def dense_parameters(self) -> Iterator[nn.Parameter]:
        r"""dense_parameters() -> Iterator[nn.Parameter]

        Return the parameters owned by the dense half.

        Returns:
            Iterator[nn.Parameter]: Dense parameters in module registration order.

        Raises:
            ValueError: If the layer was built without a dense half.
        """
        return self.require_dense().parameters()

    def split_dense_input(
        self,
        layer_input: Any,
        batch_size: int,
        num_microbatches: int,
    ) -> List[Any]:
        r"""split_dense_input(layer_input, batch_size, num_microbatches) -> List[Any]

        Split a whole-pass input for per-microbatch dense execution.

        Split-capable layers must override this method. Keeping the operation on
        the layer lets each architecture identify its batch-major fields without
        adding sparse-input dependencies to the Maglev authoring API.

        Args:
            layer_input (Any): This layer's whole-pass input.
            batch_size (int): Logical whole-pass batch size.
            num_microbatches (int): Number of dense microbatches.

        Returns:
            List[Any]: Exactly one dense input per microbatch.

        Raises:
            NotImplementedError: Unless the layer overrides this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override split_dense_input() to run "
            "with MaglevRail"
        )

    @abc.abstractmethod
    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        """The activation tuple this layer consumes; ``()`` if it consumes none."""
        ...

    @abc.abstractmethod
    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        """The activation tuple this layer produces."""
        ...

    def forward(
        self, layer_input: Any, in_activations: Activations = ()
    ) -> Activations:
        r"""forward(layer_input, in_activations=()) -> Activations

        Run the sparse half followed by the dense half.

        Args:
            layer_input: this layer's own input (its feature partition).
            in_activations: the previous layer's output activation, matching
                :meth:`in_activation_specs`. Default is ``()`` (no incoming
                activation), which is what the first layer of a model receives.

        Returns:
            Activations: this layer's output activation, matching
                :meth:`out_activation_specs`.

        Raises:
            ValueError: If the layer has no dense half and does not override this
                method.
        """
        sparse_output = None if self.sparse is None else self.sparse(layer_input)
        return self.require_dense()(sparse_output, layer_input, in_activations)


class ObservedActivationSpecsMixin:
    """Derive a Maglev layer's static contract from representative outputs."""

    _in_specs: Tuple[ActivationSpec, ...] = ()
    _out_specs: Tuple[ActivationSpec, ...] = ()

    def set_activation_specs(
        self,
        in_activations: Activations,
        out_activations: Activations,
        batch_size: Optional[int],
        in_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._in_specs = activation_specs_from_tensors(
            in_activations, batch_size, in_dtype
        )
        self._out_specs = activation_specs_from_tensors(
            out_activations, batch_size, out_dtype
        )

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return self._in_specs

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return self._out_specs


def check_layers_chain(layers: Sequence[MaglevLayer], what: str) -> None:
    """Validate that consecutive layers agree on the activation they exchange.

    Args:
        layers: the layers, in execution order.
        what: label used in the error message (e.g. ``"stage"``).

    Raises:
        ValueError: if layer ``i``'s output specs differ from layer ``i+1``'s
            input specs.
    """
    for i in range(len(layers) - 1):
        out_specs = layers[i].out_activation_specs()
        in_specs = layers[i + 1].in_activation_specs()
        if tuple(out_specs) != tuple(in_specs):
            raise ValueError(
                f"{what}: layer {i} produces {tuple(out_specs)} but layer {i + 1} "
                f"consumes {tuple(in_specs)}"
            )


class MaglevModuleList(nn.ModuleList):
    """An ordered ``ModuleList`` of Maglev layers that chains them in ``forward``.

    This is the authoring API for a Maglev model, and the single-process
    reference execution::

        acts_i = layer_i(layer_inputs[i], acts_{i-1})   with   acts_{-1} = ()

    The same authored model runs two ways:

    * **standalone** -- call it; :meth:`forward` runs every layer in one process.
      That is all this class does; it knows nothing about parallelism.
    * **pipeline-parallel** -- hand the whole model to a
      :class:`~torchrec.distributed.maglev.stage.StageWrapper`, which keeps the
      one stage a rank owns (the *same* layer modules, not copies) and is driven
      by :class:`~torchrec.distributed.maglev.pipeline.MaglevPipelineBase`.

    Both must produce identical numerics; that equivalence is what the
    correctness test checks. Container behavior (``len``, indexing, iteration) is
    inherited from ``nn.ModuleList``.

    Args:
        layers: the ordered layers, each following the :class:`MaglevLayer`
            contract.

    Raises:
        ValueError: if ``layers`` is empty, or consecutive layers disagree on the
            activation they exchange.

    Example::

        model = MaglevModuleList([layer0, layer1, layer2, layer3])
        (out,) = model([in0, in1, in2, in3])
    """

    def __init__(self, layers: Sequence[MaglevLayer]) -> None:
        if len(layers) == 0:
            raise ValueError("MaglevModuleList requires at least one layer")
        check_layers_chain(layers, "MaglevModuleList")
        super().__init__(layers)

    def preproc(self, model_input: Any) -> List[Any]:
        """Split the raw model input into one input per layer.

        This is the seam where feature partitioning / indexing lives (the Maglev
        Indexer). The MVP is a passthrough: ``model_input`` is already the list of
        per-layer inputs. Runs under ``torch.no_grad()`` (see :meth:`forward`).

        Args:
            model_input: the raw batch to partition across layers.

        Returns:
            List[Any]: one input per layer, index-aligned with the layer list.
        """
        return model_input

    def get_batch_size(self, layer_inputs: Sequence[Any]) -> int:
        """Return the batch size represented by a local stage's layer inputs.

        Called when a stage boundary uses batch-size-dependent activation specs,
        and on every pass under
        :class:`~torchrec.distributed.maglev.pipeline.MaglevRail`, which needs it
        to size the microbatch split. A Rail model must therefore override this
        whatever its specs look like.

        Override is required because layer inputs may be arbitrary structured
        values and the framework cannot infer which tensor carries the logical
        batch dimension.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses batch-size-dependent activation specs "
            "and must override get_batch_size()"
        )

    def postproc(
        self, activations: Activations, layer_input: Any
    ) -> Tuple[torch.Tensor, Any]:
        """Turn the last layer's activation into ``(losses, output)``.

        The mirror of :meth:`preproc`: where that seam splits the raw batch into
        per-layer inputs, this one closes the model -- it applies whatever head
        the architecture ends with, scores it, and returns the pair every
        TorchRec model returns::

            losses, output = model(batch)

        ``layer_input`` is the *last layer's* input, which is where the target
        lives (``ModelInput.label``, by convention). Taking the label from the
        batch rather than from a separate argument is what lets the pipeline drop
        label plumbing entirely: the last stage already holds the input it needs
        to score itself.

        Unlike :meth:`preproc`, this runs **inside** the autograd graph -- it
        computes the very loss that is backpropagated, so everything here is
        differentiated. It also runs on the last pipeline stage, inside the
        parallelized module, so a head with parameters shards with that stage.

        There is no meaningful default: a model has to say how it scores itself.
        Subclass and override.

        Args:
            activations: the last layer's output activation.
            layer_input: the last layer's input, carrying the target.

        Returns:
            Tuple[torch.Tensor, Any]: ``(losses, output)``. ``losses`` is a tensor
            the pipeline calls ``.backward()`` on, so it must be a scalar (or
            already reduced). ``output`` is whatever the model predicts -- a
            tensor for a single head, a dict or tuple for several.

        Raises:
            NotImplementedError: unless overridden.

        Example::

            class MyModel(MaglevModuleList):
                def postproc(self, activations, layer_input):
                    output = self.head(activations[0])
                    return F.mse_loss(output, layer_input.label), output
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override postproc() to return "
            "(losses, output); see MaglevModuleList.postproc"
        )

    def forward(self, model_input: Any) -> Tuple[torch.Tensor, Any]:
        """Chain the layers, threading each layer's activation into the next.

        :meth:`preproc` (run under ``torch.no_grad()``) splits ``model_input``
        into one input per layer; each layer's output activation feeds the next
        (``()`` for the first layer); :meth:`postproc` closes the model, scoring
        the last activation against the target in the last layer's input.

        Args:
            model_input: the raw batch; ``preproc`` partitions it into one input
                per layer. Inputs may be of any (per-layer) type.

        Returns:
            Tuple[torch.Tensor, Any]: ``(losses, output)``, from
            :meth:`postproc`.

        Raises:
            ValueError: if ``preproc`` does not yield one input per layer.
        """
        with torch.no_grad():
            layer_inputs = self.preproc(model_input)

        if len(layer_inputs) != len(self):
            raise ValueError(
                f"expected {len(self)} layer inputs, got {len(layer_inputs)}"
            )
        activations: Activations = ()
        for layer, layer_input in zip(self, layer_inputs):
            activations = layer(layer_input, activations)
        return self.postproc(activations, layer_inputs[-1])
