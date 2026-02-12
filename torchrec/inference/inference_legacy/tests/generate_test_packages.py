#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# @nolint

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.package import PackageExporter


class Simple(torch.nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(N, M))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.weight + input
        return output

    def set_weight(self, weight: torch.Tensor) -> None:
        self.weight[:] = torch.nn.Parameter(weight)


class Nested(torch.nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.simple = Simple(N, M)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.simple(input)


def save(name: str, model: torch.nn.Module, eg: Optional[Tuple] = None) -> None:
    with PackageExporter(str(p / name)) as e:
        e.mock("iopath.**")
        e.intern("**")
        e.save_pickle("model", "model.pkl", model)
        if eg:
            e.save_pickle("model", "example.pkl", eg)


def post_process(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


parser = argparse.ArgumentParser(description="Generate Examples")
parser.add_argument("--install_dir", help="Root directory for all output files")


def main() -> None:
    global p
    args = parser.parse_args()
    if args.install_dir is None:
        p = Path(__file__).parent / "generated"
        p.mkdir(exist_ok=True)
    else:
        p = Path(args.install_dir)

    simple = Simple(10, 20)
    post_process(simple)

    nested = Nested(10, 20)
    post_process(nested)

    save("simple", simple, (torch.rand(10, 20),))
    save("nested", nested, (torch.rand(10, 20),))


if __name__ == "__main__":
    main()  # pragma: no cover
