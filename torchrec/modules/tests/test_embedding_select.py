#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.util
import unittest
from unittest.mock import patch

import torch
from torchrec.modules.embedding_select import select_embeddings


_HAS_TRITON: bool = importlib.util.find_spec("triton") is not None


class EmbeddingSelectTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_selection_routing_and_gradients(self) -> None:
        def sorted_select(
            embeddings: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            return embeddings.index_select(0, indices)

        for index_count in [0, 3, 4, 5]:
            for use_sorted_select in [False, True]:
                for use_gather_select in [False, True]:
                    with self.subTest(
                        index_count=index_count,
                        use_sorted_select=use_sorted_select,
                        use_gather_select=use_gather_select,
                    ):
                        reverse_indices = torch.arange(index_count) % 2
                        embeddings = torch.tensor(
                            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
                        )
                        with (
                            patch(
                                "torchrec.modules.sorted_index_select.sorted_index_select",
                                side_effect=sorted_select,
                            ) as sorted_mock,
                            patch(
                                "torchrec.modules.embedding_select.torch.gather",
                                wraps=torch.gather,
                            ) as gather_mock,
                        ):
                            output = select_embeddings(
                                embeddings,
                                reverse_indices,
                                use_sorted_select=use_sorted_select,
                                use_gather_select=use_gather_select,
                            )

                        expected_sorted = use_sorted_select and (
                            not use_gather_select or index_count > 4
                        )
                        self.assertEqual(sorted_mock.call_count, int(expected_sorted))
                        self.assertEqual(
                            gather_mock.call_count,
                            int(use_gather_select and not expected_sorted),
                        )
                        torch.testing.assert_close(
                            output, embeddings.detach().index_select(0, reverse_indices)
                        )
                        output.sum().backward()
                        counts = torch.bincount(reverse_indices, minlength=2)
                        torch.testing.assert_close(
                            embeddings.grad,
                            counts[:, None].expand(2, 2).to(embeddings.dtype),
                        )
