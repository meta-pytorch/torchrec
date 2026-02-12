#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Type, TypeVar, Union

import torch
from torch.package import PackageExporter
from torchrec.inference.modules import PredictFactory

LOADER_MODULE = "__module_loader"
LOADER_FACTORY = "MODULE_FACTORY"
LOADER_CODE = f"""
import %PACKAGE%

{LOADER_FACTORY}=%PACKAGE%.%CLASS%
"""
CONFIG_MODULE = "__configs"

T = TypeVar("T")


try:
    import torch_package_importer  # @manual
except ImportError:
    pass


def load_config_text(name: str) -> str:
    return torch_package_importer.load_text("__configs", name)


def load_pickle_config(name: str, clazz: Type[T]) -> T:
    loaded_obj = torch_package_importer.load_pickle("__configs", name)
    assert isinstance(
        loaded_obj, clazz
    ), f"The loaded config {type(loaded_obj)} is not of type {clazz}"
    return loaded_obj


class PredictFactoryPackager:
    @classmethod
    @abc.abstractclassmethod
    def set_extern_modules(cls, pe: PackageExporter) -> None:
        pass

    @classmethod
    @abc.abstractclassmethod
    def set_mocked_modules(cls, pe: PackageExporter) -> None:
        pass

    @classmethod
    def save_predict_factory(
        cls,
        predict_factory: Type[PredictFactory],
        configs: Dict[str, Any],
        output: Union[str, Path, BinaryIO],
        extra_files: Dict[str, Union[str, bytes]],
        loader_code: str = LOADER_CODE,
        package_importer: Union[
            torch.package.Importer, List[torch.package.Importer]
        ] = torch.package.sys_importer,
    ) -> None:
        with PackageExporter(output, importer=package_importer) as pe:
            #  Type[PredictFactoryPackager]]` is not a function.
            cls.set_extern_modules(pe)
            #  Type[PredictFactoryPackager]]` is not a function.
            cls.set_mocked_modules(pe)
            pe.extern(["sys"])
            pe.intern("**")
            for k, v in extra_files.items():
                if isinstance(v, str):
                    pe.save_text("extra_files", k, v)
                elif isinstance(v, bytes):
                    pe.save_binary("extra_files", k, v)
                else:
                    raise ValueError(f"Unsupported type {type(v)}")
            cls._save_predict_factory(
                pe, predict_factory, configs, loader_code=loader_code
            )

    @classmethod
    def _save_predict_factory(
        cls,
        pe: PackageExporter,
        predict_factory: Type[PredictFactory],
        configs: Dict[str, Any],
        loader_code: str = LOADER_CODE,
    ) -> None:
        # If predict_factory is coming from a torch package,
        # __module__ would have <torch_package_x> prefix.
        # To save such predict factory, we need to remove
        # the prefix.
        package_name = predict_factory.__module__
        if package_name.startswith("<torch_package_"):
            package_name = ".".join(package_name.split(".")[1:])
        # Save loader entry point.
        code = loader_code.replace("%PACKAGE%", package_name).replace(
            "%CLASS%", predict_factory.__name__
        )
        pe.save_source_string(module_name=LOADER_MODULE, src=code)

        # Save configs
        for name, config in configs.items():
            if isinstance(config, str):
                pe.save_text(CONFIG_MODULE, name, config)
            else:
                pe.save_pickle(CONFIG_MODULE, name, config)

        # Save predict factory.
        pe.save_module(predict_factory.__module__)
