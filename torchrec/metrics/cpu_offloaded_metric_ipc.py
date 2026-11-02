#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transport between the trainer and the ZORM metric worker process.

Model outputs travel through `SharedUpdateRing`, a fixed set of shared-memory
slots::

    trainer                        shared memory              worker
    write slot[seq % N]     ----> [header | payload] x N  ----> read slot
    published = seq + 1     ----> published               ----> poll
    wait if seq - consumed  <---- consumed                <---- consumed = seq + 1
      >= N (ring full)

`published` and `consumed` each have a single writer and go through
`store_counter` (release) and `load_counter` (acquire), which order them
against the slot data on any CPU. Every ring generation shares them, so a
larger ring can replace the current one without losing flow control.
Everything else (layouts, ring replacements, requests, replies, update kwargs)
is pickled over `multiprocessing` pipes; `Pickler` sends tensors by value so
they never go through torch's fd-based storage sharing.
"""


import ctypes
import ctypes.util
import dataclasses
import functools
import io
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from typing import Any, Optional

import torch

ALIGNMENT_BYTES: int = 64
# Slot header: int64 seq, int64 layout_id, float64 trainer timestamp. It is
# copied together with the payload, so per-slot metadata is published with
# the data. SLOT_HEADER_BYTES is derived from the last field, so adding a field
# grows the header instead of overlapping the payload.
HEADER_INTS_END: int = 16
HEADER_TIMESTAMP_END: int = 24
SLOT_HEADER_BYTES: int = -(-HEADER_TIMESTAMP_END // ALIGNMENT_BYTES) * ALIGNMENT_BYTES
# C11 memory orders passed to libatomic.
_ATOMIC_ACQUIRE: int = 2
_ATOMIC_RELEASE: int = 3


@dataclasses.dataclass(frozen=True)
class TensorSlotSpec:
    key: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclasses.dataclass(frozen=True)
class DtypeGroup:
    """Specs sharing a dtype; one fused `_foreach_copy_` fills them."""

    dtype: torch.dtype
    spec_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SlotLayout:
    layout_id: int
    specs: tuple[TensorSlotSpec, ...]
    total_bytes: int
    groups: tuple[DtypeGroup, ...] = ()


def align(n: int) -> int:
    return (n + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES


def build_slot_layout(
    layout_id: int, tensors: list[tuple[str, torch.Tensor]]
) -> SlotLayout:
    """Specs keep the input order; offsets place each dtype group contiguously
    so a slot can be filled by one copy per group."""
    by_dtype: dict[torch.dtype, list[int]] = {}
    for index, (_, tensor) in enumerate(tensors):
        by_dtype.setdefault(tensor.dtype, []).append(index)
    offsets: dict[int, int] = {}
    groups = []
    offset = SLOT_HEADER_BYTES
    for dtype, indices in by_dtype.items():
        offset = align(offset)
        for index in indices:
            tensor = tensors[index][1]
            offsets[index] = offset
            offset += tensor.numel() * tensor.element_size()
        groups.append(DtypeGroup(dtype=dtype, spec_indices=tuple(indices)))
    specs = tuple(
        TensorSlotSpec(
            key=key,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            offset=offsets[index],
            nbytes=tensor.numel() * tensor.element_size(),
        )
        for index, (key, tensor) in enumerate(tensors)
    )
    return SlotLayout(
        layout_id=layout_id,
        specs=specs,
        total_bytes=align(offset),
        groups=tuple(groups),
    )


@functools.cache
def _libatomic() -> ctypes.CDLL:
    lib = ctypes.CDLL(ctypes.util.find_library("atomic") or "libatomic.so.1")
    load, store = lib.__atomic_load_8, lib.__atomic_store_8
    load.argtypes, load.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_uint64
    store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    store.restype = None
    return lib


def load_counter(counter: torch.Tensor) -> int:
    """Acquire-load of a shared int64 ring counter: every write made before
    the matching `store_counter` is visible after it returns."""
    return int(_libatomic().__atomic_load_8(counter.data_ptr(), _ATOMIC_ACQUIRE))


def store_counter(counter: torch.Tensor, value: int) -> None:
    """Release-store of a shared int64 ring counter (see `load_counter`)."""
    _libatomic().__atomic_store_8(counter.data_ptr(), value, _ATOMIC_RELEASE)


def header_ints(buffer: torch.Tensor) -> torch.Tensor:
    """int64 view of (seq, layout_id) at the start of a slot buffer."""
    return buffer[:HEADER_INTS_END].view(torch.int64)


def header_timestamp(buffer: torch.Tensor) -> torch.Tensor:
    return buffer[HEADER_INTS_END:HEADER_TIMESTAMP_END].view(torch.float64)


def read_header(buffer: torch.Tensor) -> tuple[int, int, float]:
    seq, layout_id = header_ints(buffer).tolist()
    return seq, layout_id, float(header_timestamp(buffer)[0])


def slot_views(buffer: torch.Tensor, layout: SlotLayout) -> list[torch.Tensor]:
    """Typed views over one slot's byte buffer, one per layout entry."""
    return [
        buffer[spec.offset : spec.offset + spec.nbytes]
        .view(spec.dtype)
        .view(spec.shape)
        for spec in layout.specs
    ]


class SharedUpdateRing:
    """Shared-memory ring of model-output slots. Picklable across `spawn` and
    over `multiprocessing` connections through torch.multiprocessing's storage
    reductions."""

    def __init__(
        self,
        num_slots: int,
        slot_bytes: int,
        counters: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        self.num_slots = num_slots
        self.slot_bytes: int = align(slot_bytes)
        self.data: torch.Tensor = torch.zeros(
            (num_slots, self.slot_bytes), dtype=torch.uint8, device="cpu"
        ).share_memory_()
        # (published, consumed): published is written only by the trainer,
        # consumed only by the worker.
        self.published, self.consumed = (
            counters
            if counters is not None
            else (
                torch.zeros(1, dtype=torch.int64, device="cpu").share_memory_(),
                torch.zeros(1, dtype=torch.int64, device="cpu").share_memory_(),
            )
        )


def _tensor_from_bytes(
    dtype: torch.dtype, shape: tuple[int, ...], payload: bytes
) -> torch.Tensor:
    if not payload:
        return torch.empty(shape, dtype=dtype)
    # clone() gives torch-allocated (aligned) storage before the dtype view.
    return (
        torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        .clone()
        .view(dtype)
        .reshape(shape)
    )


class Pickler(ForkingPickler):
    """Pickles tensors by value, except CPU tensors already in shared memory
    (the ring), which keep torch's shared-memory reduction. Results and
    kwargs therefore never create per-message shared-memory segments."""

    def reducer_override(self, obj: Any) -> Any:
        if not isinstance(obj, torch.Tensor) or (
            obj.device.type == "cpu" and obj.is_shared()
        ):
            return NotImplemented
        flat = obj.detach().to("cpu").contiguous().reshape(-1)
        payload = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
        return _tensor_from_bytes, (obj.dtype, tuple(obj.shape), payload)


def _dumps(obj: Any) -> bytes:
    buffer = io.BytesIO()
    Pickler(buffer).dump(obj)
    return buffer.getvalue()


Codec = tuple[Callable[[Any], bytes], Callable[[bytes], Any]]
_codec: Codec = (_dumps, ForkingPickler.loads)


def set_codec(codec: Codec) -> None:
    """Replaces how pipe messages are pickled in this process, e.g. so that
    classes loaded from a torch.package resolve in the other process."""
    global _codec
    _codec = codec


def send(conn: Connection, obj: Any) -> None:
    conn.send_bytes(_codec[0](obj))


def recv(conn: Connection) -> Any:
    return _codec[1](conn.recv_bytes())
