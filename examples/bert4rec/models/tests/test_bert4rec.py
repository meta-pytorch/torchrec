#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..bert4rec import BERT4Rec


class BERT4RecTest(unittest.TestCase):
    def test_bert4rec(self) -> None:
        # input tensor
        # [2, 4],
        # [3, 4, 5],
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
        self.assertEqual(
            logits.size(),
            torch.Size([input_kjt.stride(), bert4rec.max_len, bert4rec.vocab_size]),
        )
