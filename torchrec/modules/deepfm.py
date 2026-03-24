#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Deep Factorization-Machine Modules

The following modules are based off the `Deep Factorization-Machine (DeepFM) paper
<https://arxiv.org/pdf/1703.04247.pdf>`_

* Class DeepFM implements the DeepFM Framework
* Class FactorizationMachine implements FM as noted in the above paper.

"""

from typing import List

import torch
from torch import nn
from torch.fx import wrap


@wrap
def _get_flatten_input(inputs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [input.flatten(1) for input in inputs],
        dim=1,
    )


class DeepFM(nn.Module):
    r"""
    This is the `DeepFM module <https://arxiv.org/pdf/1703.04247.pdf>`_

    This module does not cover the end-end functionality of the published paper.
    Instead, it covers only the deep component of the publication. It is used to learn
    high-order feature interactions. If low-order feature interactions should
    be learnt, please use `FactorizationMachine` module instead, which will share
    the same embedding input of this module.

    To support modeling flexibility, we customize the key components as:

    * Different from the public paper, we change the input from raw sparse features to
      embeddings of the features. It allows flexibility in embedding dimensions and the
      number of embeddings, as long as all embedding tensors have the same batch size.

    * On top of the public paper, we allow users to customize the hidden layer to be any
      module, not limited to just MLP.

    The general architecture of the module is like::

                                1 x 10                  output
                                 /|\
                                  |                     pass into `dense_module`
                                  |
                                1 x 90
                                 /|\
                                  |                     concat
                                  |
                        1 x 20, 1 x 30, 1 x 40          list of embeddings

    Args:
        dense_module (nn.Module):
            any customized module that can be used (such as MLP) in DeepFM. The
            `in_features` of this module must be equal to the element counts. For
            example, if the input embedding is `[randn(3, 2, 3), randn(3, 4, 5)]`, the
            `in_features` should be: 2*3+4*5.

    Example::

        import torch
        from torchrec.fb.modules.deepfm import DeepFM
        from torchrec.fb.modules.mlp import LazyMLP
        batch_size = 3
        output_dim = 30
        # the input embedding are a torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
        ]
        dense_module = nn.Linear(192, output_dim)
        deepfm = DeepFM(dense_module=dense_module)
        deep_fm_output = deepfm(embeddings=input_embeddings)
    """

    def __init__(
        self,
        dense_module: nn.Module,
    ) -> None:
        super().__init__()
        self.dense_module = dense_module

    def forward(
        self,
        embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            embeddings (List[torch.Tensor]):
                The list of all embeddings (e.g. dense, common_sparse,
                specialized_sparse,
                embedding_features, raw_embedding_features) in the shape of::

                    (batch_size, num_embeddings, embedding_dim)

                For the ease of operation, embeddings that have the same embedding
                dimension have the option to be stacked into a single tensor. For
                example, when we have 1 trained embedding with dimension=32, 5 native
                embeddings with dimension=64, and 3 dense features with dimension=16, we
                can prepare the embeddings list to be the list of::

                    tensor(B, 1, 32) (trained_embedding with num_embeddings=1, embedding_dim=32)
                    tensor(B, 5, 64) (native_embedding with num_embeddings=5, embedding_dim=64)
                    tensor(B, 3, 16) (dense_features with num_embeddings=3, embedding_dim=32)

                .. note::
                    `batch_size` of all input tensors need to be identical.

        Returns:
            torch.Tensor: output of `dense_module` with flattened and concatenated `embeddings` as input.
        """

        # flatten each embedding to be [B, N, D] -> [B, N*D], then cat them all on dim=1
        deepfm_input = _get_flatten_input(embeddings)
        deepfm_output = self.dense_module(deepfm_input)
        return deepfm_output


class FactorizationMachine(nn.Module):
    r"""
    This is the Factorization Machine module, mentioned in the `DeepFM paper
    <https://arxiv.org/pdf/1703.04247.pdf>`_:

    This module does not cover the end-end functionality of the published paper.
    Instead, it covers only the FM part of the publication, and is used to learn
    2nd-order feature interactions.

    To support modeling flexibility, we customize the key components as different from
    the public paper:
        We change the input from raw sparse features to embeddings of the features.
        This allows flexibility in embedding dimensions and the number of embeddings,
        as long as all embedding tensors have the same batch size.

    The general architecture of the module is like::

                                1 x 10                  output
                                 /|\
                                  |                     pass into `dense_module`
                                  |
                                1 x 90
                                 /|\
                                  |                     concat
                                  |
                        1 x 20, 1 x 30, 1 x 40          list of embeddings

    Example::

        batch_size = 3
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
        ]
        fm = FactorizationMachine()
        output = fm(embeddings=input_embeddings)
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            embeddings (List[torch.Tensor]):
                The list of all embeddings (e.g. dense, common_sparse,
                specialized_sparse, embedding_features, raw_embedding_features) in the
                shape of::

                    (batch_size, num_embeddings, embedding_dim)

                For the ease of operation, embeddings that have the same embedding
                dimension have the option to be stacked into a single tensor. For
                example, when we have 1 trained embedding with dimension=32, 5 native
                embeddings with dimension=64, and 3 dense features with dimension=16, we
                can prepare the embeddings list to be the list of::

                    tensor(B, 1, 32) (trained_embedding with num_embeddings=1, embedding_dim=32)
                    tensor(B, 5, 64) (native_embedding with num_embeddings=5, embedding_dim=64)
                    tensor(B, 3, 16) (dense_features with num_embeddings=3, embedding_dim=32)

                NOTE:
                    `batch_size` of all input tensors need to be identical.

        Returns:
            torch.Tensor: output of fm with flattened and concatenated `embeddings` as input. Expected to be [B, 1].
        """

        # flatten each embedding to be [B, N, D] -> [B, N*D], then cat them all on dim=1
        fm_input = _get_flatten_input(embeddings)
        sum_of_input = torch.sum(fm_input, dim=1, keepdim=True)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        square_of_sum = sum_of_input * sum_of_input
        cross_term = square_of_sum - sum_of_square
        cross_term = torch.sum(cross_term, dim=1, keepdim=True) * 0.5  # [B, 1]
        return cross_term
