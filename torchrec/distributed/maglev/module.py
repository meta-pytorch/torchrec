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

import abc
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn

# The carrier between Maglev layers: always a tuple of tensors, empty for a
# layer with no incoming activation (the first layer of a model).
Activations = Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ActivationSpec:
    """Static description of one tensor in an inter-layer activation tuple.

    A layer declares the activations it consumes and produces as specs
    (:meth:`MaglevLayer.in_activation_specs` /
    :meth:`MaglevLayer.out_activation_specs`) so the pipeline-parallel connector
    can allocate receive buffers -- and match send/recv order -- without running
    a shape-inference forward pass.

    ``shape`` is the full per-microbatch shape, including the batch dimension.

    Args:
        shape: full shape of the tensor, batch dimension included.
        dtype: dtype of the tensor. Default is ``torch.float32``.

    Example::

        ActivationSpec(torch.Size([1024, 512]))
    """

    shape: torch.Size
    dtype: torch.dtype = torch.float32

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
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(
                f"{what}[{i}]: expected shape {tuple(spec.shape)}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != spec.dtype:
            raise ValueError(
                f"{what}[{i}]: expected dtype {spec.dtype}, got {tensor.dtype}"
            )


class MaglevLayer(abc.ABC, nn.Module):
    """Base class for a Maglev layer -- the unit of compute in a Maglev model.

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

    Args:
        None. The base class holds no state; subclasses declare their own
        constructor arguments and must implement :meth:`in_activation_specs`,
        :meth:`out_activation_specs`, and :meth:`forward`.

    Example::

        class Block(MaglevLayer):
            def __init__(self, batch_size: int, dim: int, is_first: bool) -> None:
                super().__init__()
                self._batch_size = batch_size
                self._dim = dim
                self._is_first = is_first
                self.lin: nn.Linear = nn.Linear(dim, dim)

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                if self._is_first:
                    return ()
                return (ActivationSpec(torch.Size([self._batch_size, self._dim])),)

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([self._batch_size, self._dim])),)

            def forward(self, layer_input, in_activations=()):
                x = self.lin(layer_input)
                if in_activations:
                    x = x + in_activations[0]
                return (x,)
    """

    @abc.abstractmethod
    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        """The activation tuple this layer consumes; ``()`` if it consumes none."""
        ...

    @abc.abstractmethod
    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        """The activation tuple this layer produces."""
        ...

    @abc.abstractmethod
    def forward(
        self, layer_input: Any, in_activations: Activations = ()
    ) -> Activations:
        """Run the layer over its own input and the previous layer's activation.

        Args:
            layer_input: this layer's own input (its feature partition).
            in_activations: the previous layer's output activation, matching
                :meth:`in_activation_specs`. Default is ``()`` (no incoming
                activation), which is what the first layer of a model receives.

        Returns:
            Activations: this layer's output activation, matching
            :meth:`out_activation_specs`.
        """
        ...


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
