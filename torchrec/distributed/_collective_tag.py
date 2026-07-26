#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Deterministic tag for checking whether ranks agree on a collective.

Callers pass identifying values (a label, feature keys, splits, etc.) and
get back a 31-bit tag. All ranks that run the same collective with the
same values compute the same tag. If two ranks compute different tags,
they disagree on the collective they're trying to run.
"""

import hashlib
from typing import List, Optional


# Default per-part length limit for the hash. Each part contributes at most
# this many bytes to the tag. Set high enough to cover realistic feature key
# lists on wide models while keeping per-call cost bounded.
_COLLECTIVE_TAG_MAX_BYTES: int = 32768


def _append_bytes(buf: bytearray, p: object, remaining: int) -> int:
    """Write p as bytes into buf. Stops after `remaining` bytes.

    For a str or bytes, writes up to `remaining` bytes of it.
    For a list or tuple, writes `L{count}\\x00` followed by each element
    written the same way (recursively), with each element prefixed by
    its own length so `["a,b"]` and `["a", "b"]` produce different bytes.
    For anything else, converts to str and takes up to `remaining` bytes.

    Returns the number of bytes written.
    """
    if remaining <= 0:
        return 0
    if isinstance(p, str):
        # ASCII-only: code-point slicing may overshoot `remaining` on
        # non-ASCII input. TorchRec feature keys are identifiers.
        chunk = p[:remaining].encode()
    elif isinstance(p, bytes):
        chunk = p[:remaining]
    elif isinstance(p, (list, tuple)):
        header = f"L{len(p)}\x00".encode()[:remaining]
        buf.extend(header)
        written = len(header)
        remaining -= written
        for elem in p:
            if remaining <= 0:
                break
            # Length-prefix each element to disambiguate boundaries
            # within the per-part length limit. Past-limit elements
            # may collide, e.g. at remaining=5 both "apple_pie" and
            # "apple_bar" frame as "5\x00apple".
            elem_buf = bytearray()
            _append_bytes(elem_buf, elem, remaining)
            frame = f"{len(elem_buf)}\x00".encode() + bytes(elem_buf)
            chunk = frame[:remaining]
            buf.extend(chunk)
            written += len(chunk)
            remaining -= len(chunk)
        return written
    else:
        chunk = str(p).encode()[:remaining]
    buf.extend(chunk)
    return len(chunk)


def _collective_tag_from(
    *parts: object,
    per_part_length_limits: Optional[List[Optional[int]]] = None,
) -> int:
    """Hash a list of parts into a single deterministic 31-bit tag.

    Uses hashlib.blake2b, a C-implemented hash from the standard library.
    The whole payload runs through a single native call instead of a
    Python byte loop, so cost scales with C throughput not interpreter
    overhead.

    Each part contributes at most _COLLECTIVE_TAG_MAX_BYTES bytes to the
    hash by default. Bytes past the limit are dropped and do not affect
    the tag.

    Pass per_part_length_limits (a list the same length as parts) with
    None in a slot to hash that part in full instead of capping it. Use
    this for values like splits, where any missed value would let a rank
    disagreement go undetected.

    The same parts always produce the same tag, on every rank. So if two
    ranks compute this tag and get different values, we know they disagree
    on something in the parts. Returns a non-negative value that fits in
    signed int32.
    """
    if per_part_length_limits is None:
        per_part_length_limits = [_COLLECTIVE_TAG_MAX_BYTES] * len(parts)
    assert len(per_part_length_limits) == len(parts), (
        f"per_part_length_limits length ({len(per_part_length_limits)}) "
        f"must match parts length ({len(parts)})"
    )
    hasher = hashlib.blake2b(digest_size=4)
    for i, (p, limit) in enumerate(zip(parts, per_part_length_limits)):
        if i > 0:
            hasher.update(b"\x00")
        buf = bytearray()
        # None means uncapped. Substitute a large limit so the whole
        # part gets hashed. Any realistic part is well below this.
        effective = limit if limit is not None else _UNCAPPED
        _append_bytes(buf, p, effective)
        hasher.update(bytes(buf))
    # Mask to signed int32 so the tag fits an int32 splits tensor.
    return int.from_bytes(hasher.digest(), "big") & 0x7FFFFFFF


# Substitute for "no length limit on this part." Large enough that any
# realistic tag part fits, small enough to avoid huge slice allocations.
# Any tag part bigger than this is likely a bug.
_UNCAPPED: int = 1 << 30  # 1 GiB
