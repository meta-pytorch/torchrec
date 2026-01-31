#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import functools
from typing import Any, Callable

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

"""
Prepares KJT for PT2 tracing.

KJT contains caching/lazy compute logic.
For tracing we need to drop all caches to have all compute logic in the graph.
This is done by recreation of KJT with minimal specified data.

convert_to_vb - If True recreates KJT as Variable Batch.
"""


def kjt_for_pt2_tracing(
    kjt: KeyedJaggedTensor,
    convert_to_vb: bool = False,
) -> KeyedJaggedTensor:
    # Breaking dependency cycle
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    is_vb = kjt.variable_stride_per_key()
    if convert_to_vb and not is_vb:
        stride: int = kjt.stride()
        n = len(kjt.keys())

        inverse_indices_tensor = (
            torch.arange(stride).expand(n, stride).contiguous().to(device=kjt.device())
        )
        torch._dynamo.decorators.mark_static(inverse_indices_tensor, 0)
        torch._dynamo.decorators.mark_static(inverse_indices_tensor, 1)

        lengths = kjt.lengths().long()
        # We can mark static lengths dimension as we have fixed batch_size, but using VB path for tracing
        torch._dynamo.decorators.mark_static(lengths, 0)
        values = kjt.values().long()
        torch._dynamo.decorators.mark_unbacked(values, 0)

        return KeyedJaggedTensor(
            keys=kjt.keys(),
            values=values,
            lengths=lengths,
            weights=kjt.weights_or_none(),
            stride_per_key_per_rank=torch.IntTensor([[stride]] * n, device="cpu"),
            inverse_indices=(kjt.keys(), inverse_indices_tensor),
        )

    inverse_indices = None
    stride = None

    if is_vb:
        inverse_indices = kjt.inverse_indices_or_none()

        if inverse_indices is not None:
            inverse_indices_tensor = inverse_indices[1]
            torch._dynamo.decorators.mark_static(inverse_indices_tensor, 0)
            torch._dynamo.decorators.mark_static(inverse_indices_tensor, 1)

    lengths = kjt.lengths().long()

    stride = kjt.stride()

    values = kjt.values().long()
    torch._dynamo.decorators.mark_unbacked(values, 0)
    weights = kjt.weights_or_none()
    if weights is not None:
        torch._dynamo.decorators.mark_unbacked(weights, 0)

    return KeyedJaggedTensor(
        keys=kjt.keys(),
        values=values,
        lengths=lengths,
        weights=weights,
        stride=stride if not is_vb else None,
        stride_per_key_per_rank=kjt._stride_per_key_per_rank if is_vb else None,
        inverse_indices=inverse_indices,
    )


# pyre-ignore
def default_pipeline_input_transformer(inp):
    for attr_name in ["id_list_features", "id_score_list_features"]:
        if hasattr(inp, attr_name):
            attr = getattr(inp, attr_name)
            if isinstance(attr, KeyedJaggedTensor):
                setattr(inp, attr_name, kjt_for_pt2_tracing(attr))
    return inp


def register_fake_classes() -> None:
    @torch._library.register_fake_class("fbgemm::AtomicCounter")
    class FakeAtomicCounter:
        def __init__(self, counter_):
            self.counter_ = counter_

        @classmethod
        def __obj_unflatten__(cls, flat_obj):
            return cls(**dict(flat_obj))

        def increment(self) -> int:
            self.counter_ += 1
            return self.counter_

        def decrement(self) -> int:
            self.counter_ -= 1
            return self.counter_

        def reset(self):
            self.counter_ = 0

        def get(self) -> int:
            return self.counter_

        def set(self, val):
            self.counter_ = val

    @torch._library.register_fake_class("fbgemm::TensorQueue")
    class FakeTensorQueue:
        def __init__(self, queue, init_tensor):
            self.queue = queue
            self.init_tensor = init_tensor

        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            return cls(**dict(flattened_ctx))

        def push(self, x):
            self.queue.append(x)

        def pop(self):
            if len(self.queue) == 0:
                return self.init_tensor
            return self.queue.pop(0)

        def top(self):
            if len(self.queue) == 0:
                return self.init_tensor
            return self.queue[0]

        def size(self):
            return len(self.queue)


def deregister_fake_classes() -> None:
    torch._library.fake_class_registry.deregister_fake_class("fbgemm::AtomicCounter")
    torch._library.fake_class_registry.deregister_fake_class("fbgemm::TensorQueue")


# pyre-ignore[24]
def pt2_compile_callable(f: Callable) -> Callable:
    """
    This method is used to decorate the update and compute methods of a metric computation class.
    If the metric computation class has enable_pt2_compile attribute set to True,
    then the update and compute methods will be compiled using torch.compile.
    """

    @functools.wraps(f)
    # pyre-ignore[3]
    def inner_forward(
        ref: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(ref, "enable_pt2_compile") and ref.enable_pt2_compile:
            pt2_compiled_attr_name = f"_{f.__name__}_pt2_compiled"
            if not hasattr(ref, pt2_compiled_attr_name):
                setattr(ref, pt2_compiled_attr_name, torch.compile(f))
            return getattr(ref, pt2_compiled_attr_name)(ref, *args, **kwargs)

        return f(ref, *args, **kwargs)

    return inner_forward
