#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


PROPOGATE_DEVICE: bool = False


def set_propogate_device(val: bool) -> None:
    global PROPOGATE_DEVICE
    PROPOGATE_DEVICE = val


def get_propogate_device() -> bool:
    global PROPOGATE_DEVICE
    return PROPOGATE_DEVICE
