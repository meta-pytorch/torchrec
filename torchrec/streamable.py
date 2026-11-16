#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import abc

import torch


class Multistreamable(abc.ABC):
    """
    Objects implementing this interface are allowed to be transferred
    from one CUDA stream to another.
    torch.Tensor and (Keyed)JaggedTensor implement this interface.
    """

    @abc.abstractmethod
    def record_stream(self, stream: torch.Stream) -> None:
        """
        See https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
        """
        ...


class Pipelineable(Multistreamable):
    """
    This interface contains two methods, one for moving an input across devices,
    the other one for marking streams that operate the input.

    torch.Tensor implements this interface and we can used it in many applications.
    Another example is torchrec.(Keyed)JaggedTensor, which we use as the input to
    torchrec.EmbeddingBagCollection, which in turn is often the first layer of many models.
    Some models take compound inputs, which should implement this interface.
    """

    @abc.abstractmethod
    def to(self, device: torch.device, non_blocking: bool) -> "Pipelineable":
        """
        Please be aware that according to https://pytorch.org/docs/stable/generated/torch.Tensor.to.html,
        `to` might return self or a copy of self.  So please remember to use `to` with the assignment operator,
        for example, `in = in.to(new_device)`.
        """
        ...
