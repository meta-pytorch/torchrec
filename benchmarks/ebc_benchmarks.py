#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import sys
from typing import List, Tuple

# pyrefly: ignore[missing-import]
import ebc_benchmarks_utils
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import EmbeddingLocation
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection

# Reference: https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD
DLRM_NUM_EMBEDDINGS_PER_FEATURE = [
    45833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]


def get_shrunk_dlrm_num_embeddings(reduction_degree: int) -> List[int]:
    return [
        num_emb if num_emb < 10000000 else int(num_emb / reduction_degree)
        for num_emb in DLRM_NUM_EMBEDDINGS_PER_FEATURE
    ]


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.mode == "ebc_comparison_dlrm":
        print("Running EBC vs. FusedEBC on DLRM EMB")

        for reduction_degree in [128, 64, 32]:
            embedding_bag_configs: List[EmbeddingBagConfig] = [
                EmbeddingBagConfig(
                    name=f"ebc_{idx}",
                    embedding_dim=128,
                    num_embeddings=num_embeddings,
                    feature_names=[f"ebc_{idx}_feat_1"],
                )
                for idx, num_embeddings in enumerate(
                    get_shrunk_dlrm_num_embeddings(reduction_degree)
                )
            ]
            (
                ebc_time_avg,
                ebc_time_std,
                fused_ebc_time_avg,
                fused_ebc_time_std,
                speedup,
            ) = get_ebc_comparison(embedding_bag_configs, device)

            print(f"when DLRM EMB is reduced by {reduction_degree} times:")
            print(f"ebc_time = {ebc_time_avg} +/- {ebc_time_std} sec")
            print(f"fused_ebc_time = {fused_ebc_time_avg} +/- {fused_ebc_time_std} sec")
            print(f"speedup = {speedup}")

    elif args.mode == "fused_ebc_uvm":
        print("Running DLRM EMB on FusedEBC with UVM/UVM-caching")
        embedding_bag_configs: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                name=f"ebc_{idx}",
                embedding_dim=128,
                num_embeddings=num_embeddings,
                feature_names=[f"ebc_{idx}_feat_1"],
            )
            for idx, num_embeddings in enumerate(get_shrunk_dlrm_num_embeddings(2))
        ]
        fused_ebc_time_avg, fused_ebc_time_std = get_fused_ebc_uvm_time(
            embedding_bag_configs, device, EmbeddingLocation.MANAGED_CACHING
        )
        print(
            f"FusedEBC with UVM caching on DLRM: {fused_ebc_time_avg} +/- {fused_ebc_time_std} sec"
        )

        embedding_bag_configs: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                name=f"ebc_{idx}",
                embedding_dim=128,
                num_embeddings=num_embeddings,
                feature_names=[f"ebc_{idx}_feat_1"],
            )
            for idx, num_embeddings in enumerate(DLRM_NUM_EMBEDDINGS_PER_FEATURE)
        ]
        fused_ebc_time_avg, fused_ebc_time_std = get_fused_ebc_uvm_time(
            embedding_bag_configs, device, EmbeddingLocation.MANAGED
        )
        print(
            f"FusedEBC with UVM management on DLRM: {fused_ebc_time_avg} plus/minus {fused_ebc_time_std} sec"
        )

    elif args.mode == "ebc_comparison_scaling":
        print("Running EBC vs. FusedEBC scaling experiment")

        num_tables_list = [10, 100, 1000]
        embedding_dim_list = [4, 8, 16, 32, 64, 128]
        num_embeddings_list = [4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192]

        for num_tables in num_tables_list:
            for num_embeddings in num_embeddings_list:
                for embedding_dim in embedding_dim_list:
                    embedding_bag_configs: List[EmbeddingBagConfig] = [
                        EmbeddingBagConfig(
                            name=f"ebc_{idx}",
                            embedding_dim=embedding_dim,
                            num_embeddings=num_embeddings,
                            feature_names=[f"ebc_{idx}_feat_1"],
                        )
                        for idx in range(num_tables)
                    ]
                    ebc_time, _, fused_ebc_time, _, speedup = get_ebc_comparison(
                        embedding_bag_configs, device, epochs=3
                    )
                    print(
                        f"EBC num_tables = {num_tables}, num_embeddings = {num_embeddings}, embedding_dim = {embedding_dim}:"
                    )
                    print(
                        f"ebc_time = {ebc_time} sec, fused_ebc_time = {fused_ebc_time} sec, speedup = {speedup}"
                    )


def get_fused_ebc_uvm_time(
    embedding_bag_configs: List[EmbeddingBagConfig],
    device: torch.device,
    location: EmbeddingLocation,
    epochs: int = 100,
) -> Tuple[float, float]:

    fused_ebc = FusedEmbeddingBagCollection(
        tables=embedding_bag_configs,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.02},
        device=device,
        location=location,
    )

    dataset = ebc_benchmarks_utils.get_random_dataset(
        batch_size=64,
        num_batches=10,
        num_dense_features=1024,
        embedding_bag_configs=embedding_bag_configs,
    )

    fused_ebc_time_avg, fused_ebc_time_std = ebc_benchmarks_utils.train(
        model=fused_ebc,
        optimizer=None,
        dataset=dataset,
        device=device,
        epochs=epochs,
    )

    return fused_ebc_time_avg, fused_ebc_time_std


def get_ebc_comparison(
    embedding_bag_configs: List[EmbeddingBagConfig],
    device: torch.device,
    epochs: int = 100,
) -> Tuple[float, float, float, float, float]:

    # Simple EBC module wrapping a list of nn.EmbeddingBag
    ebc = EmbeddingBagCollection(
        tables=embedding_bag_configs,
        device=device,
    )
    optimizer = torch.optim.SGD(ebc.parameters(), lr=0.02)

    # EBC with fused optimizer backed by fbgemm SplitTableBatchedEmbeddingBagsCodegen
    fused_ebc = FusedEmbeddingBagCollection(
        tables=embedding_bag_configs,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.02},
        device=device,
    )

    dataset = ebc_benchmarks_utils.get_random_dataset(
        batch_size=64,
        num_batches=10,
        num_dense_features=1024,
        embedding_bag_configs=embedding_bag_configs,
    )

    ebc_time_avg, ebc_time_std = ebc_benchmarks_utils.train(
        model=ebc,
        optimizer=optimizer,
        dataset=dataset,
        device=device,
        epochs=epochs,
    )
    fused_ebc_time_avg, fused_ebc_time_std = ebc_benchmarks_utils.train(
        model=fused_ebc,
        optimizer=None,
        dataset=dataset,
        device=device,
        epochs=epochs,
    )
    speedup = ebc_time_avg / fused_ebc_time_avg

    return ebc_time_avg, ebc_time_std, fused_ebc_time_avg, fused_ebc_time_std, speedup


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchRec ebc benchmarks")
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        default=False,
        help="specify whether to use cpu",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ebc_comparison_dlrm",
        help="specify 'ebc_comparison_dlrm', 'ebc_comparison_scaling' or 'fused_ebc_uvm'",
    )
    return parser.parse_args(argv)


def invoke_main() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
