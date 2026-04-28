#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import gzip
import json
import logging
import os
import pickle
import platform
from typing import Any

import torch

logger: logging.Logger = logging.getLogger(__name__)


def get_gpu_type() -> str:
    """Return the GPU device name, or 'N/A' if CUDA is unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


def get_cpu_type() -> str:
    """Return the CPU model string."""
    return platform.processor() or platform.machine()


def create_trace_file_name(profile_name: str, rank: int) -> str:
    """Create a unique trace file name for the given rank and profile name."""
    return f"trace-{profile_name}-rank{rank}.json.gz"


def create_snapshot_file_name(profile_name: str, rank: int) -> str:
    """Create a unique memory snapshot file name for the given rank and profile name."""
    return f"memory-{profile_name}-rank{rank}.pickle"


def _load_trace_events(trace_path: str) -> list[dict[str, Any]]:
    if trace_path.endswith(".gz"):
        with gzip.open(trace_path, "rt") as f:
            trace_data = json.load(f)
    else:
        with open(trace_path, "r") as f:
            trace_data = json.load(f)

    # pyre-ignore[6]: json.load returns Any
    if isinstance(trace_data, list):
        return trace_data
    if isinstance(trace_data, dict):
        return trace_data.get("traceEvents", [])
    return []


def _extract_stream_tracks(
    events: list[dict[str, Any]],
) -> dict[tuple[int, int], str]:
    """Extract GPU stream track names from chrome trace metadata events.

    In the Chrome Trace Format each event has a ``"ph"`` (phase) field:
      - ``"M"``  — metadata event, used to set process/thread names
      - ``"X"``  — complete duration event (has ``ts`` and ``dur``)
      - ``"B"``/``"E"`` — begin/end pair for duration events

    Metadata events with ``"name": "thread_name"`` assign a human-readable
    name (stored in ``args.name``) to a ``(pid, tid)`` pair. This function
    collects those whose name starts with ``"stream"`` (e.g. ``"stream 7"``),
    which represent CUDA stream tracks in PyTorch profiler traces.
    """
    track_names: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "thread_name":
            continue
        pid = event.get("pid", 0)
        tid = event.get("tid", 0)
        args = event.get("args", {})
        if not (
            isinstance(args, dict) and isinstance(pid, int) and isinstance(tid, int)
        ):
            continue
        name = args.get("name", "")
        if isinstance(name, str) and name.startswith("stream"):
            track_names[(pid, tid)] = name
    return track_names


def _merged_active_time(intervals: list[tuple[float, float]]) -> float:
    """Merge overlapping time intervals and return total active duration.

    Sorts intervals by start time, then merges any that overlap or are
    adjacent. The result is the sum of all merged interval lengths, which
    represents the total wall-clock time covered without double-counting.

    Example::

        >>> _merged_active_time([(0, 5), (3, 8), (10, 15)])
        13.0  # merged to [(0, 8), (10, 15)] -> 8 + 5 = 13

        >>> _merged_active_time([(0, 10), (2, 4), (6, 12)])
        12.0  # fully overlapping -> [(0, 12)] -> 12
    """
    if not intervals:
        return 0.0
    intervals.sort()
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        _, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (merged[-1][0], max(prev_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def parse_chrome_trace_gpu_utilization(trace_path: str) -> dict[str, float]:
    """Parse a perfetto/chrome trace and return per-stream GPU utilization.

    GPU utilization for each stream is the fraction of the total trace duration
    during which the stream has active events. Stream tracks are identified by
    thread-name metadata events whose name starts with ``"stream"``.

    Args:
        trace_path: Path to a chrome trace file (``.json`` or ``.json.gz``).

    Returns:
        Mapping of stream name to utilization ratio (0.0 to 1.0).
    """
    events = _load_trace_events(trace_path)
    stream_tracks = _extract_stream_tracks(events)
    if not stream_tracks:
        return {}

    stream_intervals: dict[str, list[tuple[float, float]]] = {
        name: [] for name in stream_tracks.values()
    }
    global_min_ts: float = float("inf")
    global_max_ts: float = float("-inf")

    for event in events:
        ts = event.get("ts")
        dur = event.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        if dur <= 0:
            continue
        pid = event.get("pid", 0)
        tid = event.get("tid", 0)
        if not isinstance(pid, int) or not isinstance(tid, int):
            continue

        end = float(ts) + float(dur)
        key = (pid, tid)
        if key in stream_tracks:
            stream_intervals[stream_tracks[key]].append((float(ts), end))
        global_min_ts = min(global_min_ts, float(ts))
        global_max_ts = max(global_max_ts, end)

    total_duration = global_max_ts - global_min_ts
    if total_duration <= 0:
        return {}

    return {
        name: _merged_active_time(stream_intervals[name]) / total_duration
        for name in sorted(stream_intervals.keys())
    }


def _merge_gpu_utilization_metrics(
    trace_path: str,
    metrics: dict[str, object],
) -> None:
    gpu_utilization = parse_chrome_trace_gpu_utilization(trace_path)
    for stream_name, util in gpu_utilization.items():
        key = stream_name.replace(" ", "_") + "_utilization"
        metrics[key] = util


def parse_memory_snapshot_peak_per_stream(
    snapshot_path: str,
    device: int = 0,
) -> dict[int, float]:
    """Parse a PyTorch memory snapshot and return peak memory usage per stream.

    Replays alloc/free events chronologically, tracking current memory per
    CUDA stream and recording the high-water mark for each.

    Args:
        snapshot_path: Path to the ``.pickle`` memory snapshot file.
        device: Device index in ``device_traces`` (default 0).

    Returns:
        Mapping of stream ID to peak memory in MB.
    """
    with open(snapshot_path, "rb") as f:
        data = pickle.load(f)

    device_traces = data.get("device_traces", [])
    if not isinstance(device_traces, list) or device >= len(device_traces):
        return {}

    traces = device_traces[device]
    if not isinstance(traces, list):
        return {}

    active_allocs: dict[int, tuple[int, int]] = {}
    current_per_stream: dict[int, int] = {}
    peak_per_stream: dict[int, int] = {}

    for event in traces:
        action = event.get("action", "")
        addr = event.get("addr", 0)
        if not isinstance(addr, int):
            addr = int(addr)

        if action == "alloc":
            size = event.get("size", 0)
            stream = event.get("stream", 0)
            if not isinstance(size, int):
                size = int(size)
            if not isinstance(stream, int):
                stream = int(stream)
            active_allocs[addr] = (stream, size)
            current = current_per_stream.get(stream, 0) + size
            current_per_stream[stream] = current
            if current > peak_per_stream.get(stream, 0):
                peak_per_stream[stream] = current

        elif action in ("free_requested", "free_completed"):
            if addr in active_allocs:
                stream, size = active_allocs.pop(addr)
                current_per_stream[stream] = current_per_stream.get(stream, 0) - size

    return {
        stream: bytes_val / (1024 * 1024)
        for stream, bytes_val in peak_per_stream.items()
    }


def dump_benchmark_result(
    result: Any,
    output_dir: str,
    world_size: int,
) -> None:
    """Write benchmark result to a JSON file in *output_dir*.

    The file is named ``torchrec_benchmark_<short_name>_<rank>.json`` and
    contains all metrics from ``to_dict()``, plus hardware and source info.

    Args:
        result: A ``BenchmarkResult`` instance (typed as ``Any`` to avoid a
            circular dependency with ``base``).
        output_dir: Directory where the JSON file is written.
        world_size: Number of ranks in the distributed run.
    """
    data: dict[str, object] = {
        "short_name": result.short_name,
        "rank": result.rank,
        "world_size": world_size,
        "gpu_type": get_gpu_type(),
        "cpu_type": get_cpu_type(),
        "metrics": result.to_dict(),
    }

    trace_path = os.path.join(
        output_dir,
        create_trace_file_name(result.short_name, result.rank),
    )
    metrics = data["metrics"]
    if os.path.exists(trace_path) and isinstance(metrics, dict):
        try:
            _merge_gpu_utilization_metrics(trace_path, metrics)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse chrome trace for GPU utilization: {e}")

    snapshot_path = os.path.join(
        output_dir,
        create_snapshot_file_name(result.short_name, result.rank),
    )
    if os.path.exists(snapshot_path):
        try:
            peak_mem = parse_memory_snapshot_peak_per_stream(snapshot_path)
            metrics = data["metrics"]
            assert isinstance(metrics, dict)
            for stream_id, peak_mb in sorted(peak_mem.items()):
                key = f"stream_{stream_id}_peak_memory_mb"
                metrics[key] = round(peak_mb, 2)
        except Exception as e:
            logger.warning(f"Failed to parse memory snapshot: {e}")

    path = os.path.join(
        output_dir,
        f"torchrec_benchmark_{result.short_name}_{result.rank}.json",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    logger.info(f"Benchmark result written to {path}")
