#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Split a CUDA memory snapshot pickle file by stream ID.

Usage:
    # Split a single snapshot file
    python split_snapshot.py memory-base_pipeline_light-rank0.pickle

    # Split multiple snapshot files
    python split_snapshot.py snapshot1.pickle snapshot2.pickle

    # Split all pickle files in current directory
    python split_snapshot.py *.pickle

Output:
    For each input file, creates one pickle file per unique stream ID:
    - memory-base_pipeline_light-rank0.pickle -> memory-base_pipeline_light-rank0_0.pickle
                                              -> memory-base_pipeline_light-rank0_7.pickle
                                              -> memory-base_pipeline_light-rank0_none.pickle

Data structure of the snapshot pickle file:
>>> import pickle
>>> with open('memory-base_pipeline_light-rank0.pickle', 'rb') as f:
...     data=pickle.load(f)
...
>>> data.keys()
dict_keys(['segments', 'device_traces', 'allocator_settings', 'external_annotations'])

>>> data['segments'][0].keys()
dict_keys(['device', 'address', 'total_size', 'allocated_size', 'active_size', 'requested_size', 'stream', 'segment_type', 'segment_pool_id', 'is_expandable', 'frames', 'blocks'])

>>> type(data['device_traces'])
<class 'list'>
>>> data['device_traces'][1:]
[[], [], [], [], [], [], []]
>>> type(data['device_traces'][0])
<class 'list'>
>>> data['device_traces'][0][0].keys()
dict_keys(['action', 'addr', 'size', 'stream', 'time_us', 'compile_context', 'user_metadata', 'frames'])
>>> data['device_traces'][0][0]['stream']
140303324794416

"""

import argparse
import os
import pickle
from typing import Any, Dict, List, Set


def get_unique_streams(
    device_traces: List[List[Dict[str, Any]]], segments: List[Dict[str, Any]]
) -> Set[int]:
    """Extract all unique stream IDs from device_traces and segments."""
    streams = set()
    for device_trace_list in device_traces:
        for trace in device_trace_list:
            stream_id = trace.get("stream")
            streams.add(stream_id)
    for segment in segments:
        stream_id = segment.get("stream")
        streams.add(stream_id)
    return streams


def filter_device_traces_by_stream(
    device_traces: List[List[Dict[str, Any]]], stream_id: int
) -> List[List[Dict[str, Any]]]:
    """Filter device_traces to only include traces with the specified stream ID."""
    filtered_traces = []
    for device_trace_list in device_traces:
        filtered_list = [
            trace for trace in device_trace_list if trace.get("stream") == stream_id
        ]
        filtered_traces.append(filtered_list)
    return filtered_traces


def filter_segments_by_stream(
    segments: List[Dict[str, Any]], stream_id: int
) -> List[Dict[str, Any]]:
    """Filter segments to only include those with the specified stream ID."""
    return [segment for segment in segments if segment.get("stream") == stream_id]


def split_snapshot(input_path: str) -> None:
    """Split a snapshot pickle file by stream ID."""
    print(f"Processing: {input_path}")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    device_traces = data.get("device_traces", [])
    segments = data.get("segments", [])
    unique_streams = get_unique_streams(device_traces, segments)

    print(f"  Found {len(unique_streams)} unique streams: {sorted(unique_streams)}")

    base_path, ext = os.path.splitext(input_path)

    for stream_id in sorted(unique_streams):
        filtered_traces = filter_device_traces_by_stream(device_traces, stream_id)
        filtered_segments = filter_segments_by_stream(segments, stream_id)

        split_data = {
            key: value
            for key, value in data.items()
            if key not in ("device_traces", "segments")
        }
        split_data["device_traces"] = filtered_traces
        split_data["segments"] = filtered_segments

        stream_label = stream_id if stream_id is not None else "none"
        output_path = f"{base_path}_{stream_label}{ext}"

        with open(output_path, "wb") as f:
            pickle.dump(split_data, f)

        trace_count = sum(len(traces) for traces in filtered_traces)
        segment_count = len(filtered_segments)
        print(
            f"  Written: {output_path} ({trace_count} traces, {segment_count} segments)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split CUDA memory snapshot pickle files by stream ID."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more snapshot.pickle files to split",
    )
    args = parser.parse_args()

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        split_snapshot(file_path)


if __name__ == "__main__":
    main()
