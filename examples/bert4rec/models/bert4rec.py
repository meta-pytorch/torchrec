#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Clone the module to N copies

    Args:
        module (nn.Module): module to clone
        N (int): number of copies

    Returns:
        nn.ModuleList of module copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention

    Args:
        query (torch.Tensor): the query tensor
        key (torch.Tensor): the key tensor
        value (torch.Tensor): the value tensor
        mask (torch.Tensor): the mask tensor
        dropout (nn.Dropout): the dropout layer

    Example::

        self.attention = Attention()
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward function

        Args:
            query (torch.Tensor): the query tensor
            key (torch.Tensor): the key tensor
            value (torch.Tensor): the value tensor
            mask (torch.Tensor): the mask tensor
            dropout (nn.Dropout): the dropout layer

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = nn.functional.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.

    Args:
        num_heads (int): number of attention heads
        dim_model (int): input/output dimensionality
        dropout (float): the dropout probability
        mask (torch.Tensor): the mask tensor
        device: (Optional[torch.device]).

    Example::

        self.attention = MultiHeadedAttention(
            num_heads=attn_heads, dim_model=hidden, dropout=dropout, device=device
        )
        self.attention.forward(query, key, value, mask=mask)
    """

    def __init__(
        self,
        num_heads: int,
        dim_model: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert dim_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k: int = dim_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(dim_model, dim_model, device=device) for _ in range(3)]
        )
        self.output_linear = nn.Linear(dim_model, dim_model, device=device)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        forward function

        Args:
            query (torch.Tensor): the query tensor
            key (torch.Tensor): the key tensor
            value (torch.Tensor): the value tensor
            mask (torch.Tensor): the mask tensor

        Returns:
            torch.Tensor.
        """
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from dim_model => num_heads x d_k
        # self.linear[0, 1, 2] is query weight matrix, key weight matrix, and
        # value weight matrix, respectively.
        # l(x) represents the transformed query matrix, key matrix and value matrix
        # l(x) has shape (batch_size, seq_len, dim_model). You can think l(x) as
        # the matrices from a one-head attention; or you can think
        # l(x).view(...).transpose(...) as the matrices of num_heads attentions,
        # each attention has d_k dimension.
        query, key, value = [
            linearLayer(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
            for linearLayer, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: batch_size, num_heads, seq_len, d_k
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # each attention's output is d_k dimension. Concat num_heads attention's outputs
        # x shape: batch_size, seq_len, dim_model
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward block.

    Args:
        dim_model (int): input/output dimensionality
        d_ff (int): hidden dimensionality
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.feed_forward = PositionwiseFeedForward(
            dim_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, device=device
        )
    """

    def __init__(
        self,
        dim_model: int,
        d_ff: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(dim_model, d_ff, device=device)
        self.w_2 = nn.Linear(d_ff, dim_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function including the first Linear layer, ReLu, Dropout
        and the final Linear layer

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor.
        """
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.

    Args:
        size (int): layerNorm size
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.input_sublayer = SublayerConnection(
            size=hidden, dropout=dropout, device=device
        )
    """

    def __init__(
        self,
        size: int,
        dropout: float,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        forward function including the norm layer, sublayer and dropout, finally
        add up with the input tensor

        Args:
            x (torch.Tensor): the input tensor
            sublayer (Callable[[torch.Tensor): callable layer

        Returns:
            torch.Tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection

    Args:
        hidden (int): hidden size of transformer
        attn_heads (int): head sizes of multi-head attention
        feed_forward_hidden (int): feed_forward_hidden, usually 4*hidden_size
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(emb_dim, nhead, emb_dim * 4, dropout, device=device)
                for _ in range(num_layers)
            ]
        )
    """

    def __init__(
        self,
        hidden: int,
        attn_heads: int,
        feed_forward_hidden: int,
        dropout: float,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadedAttention(
            num_heads=attn_heads, dim_model=hidden, dropout=dropout, device=device
        )
        self.feed_forward = PositionwiseFeedForward(
            dim_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, device=device
        )
        self.input_sublayer = SublayerConnection(
            size=hidden, dropout=dropout, device=device
        )
        self.output_sublayer = SublayerConnection(
            size=hidden, dropout=dropout, device=device
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        forward function

        Args:
            x (torch.Tensor): the input tensor
            mask (torch.BoolTensor): determine which position has been masked

        Returns:
            torch.Tensor.
        """
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class HistoryArch(torch.nn.Module):
    """
    embedding.HistoryArch is the input embedding layer the BERT4Rec model
    as described in Section 3.4 of the paper.  It consits of an item
    embedding table and a positional embedding table.

    The item embedding table consists of vocab_size vectors.  The
    positional embedding table has history_len vectors.  Both
    kinds of embedding vectors have length emb_dim.  As mentioned
    in Section 3.7, BERT4Rec differs from BERT lacking the
    sentence embedding.

    Note that for the item embedding table, we have applied TorchRec
    EmbeddingCollection which supports sharding

    Args:
        vocab_size (int): the item count including mask and padding
        history_len (int): the max length
        emb_dim (int): embedding dimension
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.history = HistoryArch(
            vocab_size, max_len, emb_dim, dropout=dropout, device=device
        )
        x = self.history(input)
    """

    def __init__(
        self,
        vocab_size: int,
        history_len: int,
        emb_dim: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.history_len = history_len
        self.positional = nn.Parameter(torch.randn(history_len, emb_dim, device=device))
        self.layernorm = torch.nn.LayerNorm([history_len, emb_dim], device=device)
        self.dropout = torch.nn.Dropout(p=dropout)

        item_embedding_config = EmbeddingConfig(
            name="item_embedding",
            embedding_dim=emb_dim,
            num_embeddings=vocab_size,
            feature_names=["item"],
            weight_init_max=1.0,
            weight_init_min=-1.0,
        )
        self.ec = EmbeddingCollection(
            tables=[item_embedding_config],
            device=device,
        )

    def forward(self, id_list_features: KeyedJaggedTensor) -> torch.Tensor:
        """
        forward function: first query the item embedding and do the padding
        then add up with positional parameters and do the norm and dropout

        Args:
            id_list_features (KeyedJaggedTensor): the input KeyedJaggedTensor

        Returns:
            torch.Tensor.
        """
        jt_dict = self.ec(id_list_features)

        padded_embeddings = [
            torch.ops.fbgemm.jagged_2d_to_dense(
                values=jt_dict[e].values(),
                offsets=jt_dict[e].offsets(),
                max_sequence_length=self.history_len,
            ).view(-1, self.history_len, self.emb_dim)
            for e in id_list_features.keys()
        ]
        item_output = torch.cat(
            padded_embeddings,
            dim=1,
        )
        batch_size = id_list_features.stride()
        positional_output = self.positional.unsqueeze(0).repeat(batch_size, 1, 1)
        x = item_output + positional_output
        return self.dropout(self.layernorm(x))


class BERT4Rec(nn.Module):
    """
    The overall arch described in the BERT4Rec paper: (https://arxiv.org/abs/1904.06690)
    the encoder_layer was described in the section of 3.3, the output_layer was described in the
    section of 3.5

    Args:
        vocab_size (int): the item count including mask and padding
        max_len (int): the max length
        emb_dim (int): embedding dimension
        nhead (int): number of the transformation headers
        num_layers (int): number of the transformation layers
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        input_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["item"],
            values=torch.tensor([2, 4, 3, 4, 5]),
            lengths=torch.tensor([2, 3]),
        )
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        input_kjt = input_kjt.to(device)
        bert4rec = BERT4Rec(
            vocab_size=6, max_len=3, emb_dim=4, nhead=4, num_layers=4, device=device
        )
        logits = bert4rec(input_kjt)
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        emb_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.history = HistoryArch(
            vocab_size, max_len, emb_dim, dropout=dropout, device=device
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(emb_dim, nhead, emb_dim * 4, dropout, device=device)
                for _ in range(num_layers)
            ]
        )
        # this linear layer is different from the original paper which used the
        # embedding layer weight to get the final output by matmul, from our experimentation,
        # this implementation has higher performance
        self.out = nn.Linear(self.emb_dim, self.vocab_size, device=device)

    def forward(self, input: KeyedJaggedTensor) -> torch.Tensor:
        """
        forward function: first get the item embedding result and
        fit into transformer blocks and fit into the last linaer
        layer to get the final output

        Args:
            input (KeyedJaggedTensor): the input KeyedJaggedTensor

        Returns:
            torch.Tensor.
        """
        dense_tensor = input["item"].to_padded_dense(desired_length=self.max_len)
        mask = (
            (dense_tensor > 0)
            .unsqueeze(1)
            .repeat(1, dense_tensor.size(1), 1)
            .unsqueeze(1)
        )
        # embedding the indexed sequence to sequence of vectors
        x = self.history(input)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return self.out(x)
