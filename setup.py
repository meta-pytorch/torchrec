#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
            version = f.readline().strip()
        if sha is not None and "OFFICIAL_RELEASE" not in os.environ:
            version += "+" + sha[:7]

    if sha is None:
        sha = "Unknown"
    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "torchrec" / "version.py"
    with open(version_path, "w") as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec setup")
    return parser.parse_known_args(argv)


def main(argv: List[str]) -> None:
    args, unknown = parse_args(argv)

    with open(
        os.path.join(os.path.dirname(__file__), "README.MD"), encoding="utf8"
    ) as f:
        readme = f.read()
    with open(
        os.path.join(os.path.dirname(__file__), "install-requirements.txt"),
        encoding="utf8",
    ) as f:
        reqs = f.read()
        install_requires = reqs.strip().split("\n")

    version, sha = _get_version()
    _export_version(version, sha)

    print(f"-- torchrec building version: {version}")

    packages = find_packages(
        exclude=(
            "*tests",
            "*test",
            "examples",
            "*examples.*",
            "*benchmarks",
            "*build",
            "*rfc",
        )
    )
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name="torchrec",
        version=version,
        author="TorchRec Team",
        author_email="packages@pytorch.org",
        maintainer="TroyGarden",
        maintainer_email="hhy@meta.com",
        description="TorchRec: Pytorch library for recommendation systems",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/meta-pytorch/torchrec",
        license="BSD-3",
        keywords=[
            "pytorch",
            "recommendation systems",
            "sharding",
            "distributed training",
        ],
        python_requires=">=3.10",
        install_requires=install_requires,
        packages=packages,
        zip_safe=False,
        # PyPI package information.
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main(sys.argv[1:])
