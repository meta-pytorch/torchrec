# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utility functions for Torch TPU Embedding modules."""

import torch


def get_device() -> torch.device:
  """Returns the TPU PyTorch device."""
  return torch.device("tpu")


def get_compile_count() -> int:
  """Returns total PyTorch Dynamo frame compile count."""
  return sum(torch._dynamo.convert_frame.FRAME_COMPILE_COUNTER.values())
