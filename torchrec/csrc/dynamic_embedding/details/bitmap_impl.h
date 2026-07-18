/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <stdint.h>
#include <torchrec/csrc/dynamic_embedding/details/bits_op.h>

namespace torchrec {

template <typename T>
inline Bitmap<T>::Bitmap(int64_t num_bits)
    : num_total_bits_(num_bits),
      num_values_((num_bits + num_bits_per_value - 1) / num_bits_per_value),
      values_(new T[num_values_]),
      next_free_bit_(0) {
  std::fill(values_.get(), values_.get() + num_values_, -1);
}

template <typename T>
inline int64_t Bitmap<T>::next_free_bit() {
  int64_t result = next_free_bit_;
  int64_t offset = result / num_bits_per_value;
  T value = values_[offset];
  // set the last 1 bit to zero
  values_[offset] = value & (value - 1);
  while (values_[offset] == 0 && offset < num_values_) {
    offset++;
  }
  value = values_[offset];
  if (value) [[likely]] {
    next_free_bit_ = offset * num_bits_per_value + ctz(value);
  } else {
    next_free_bit_ = num_total_bits_;
  }

  return result;
}

template <typename T>
inline void Bitmap<T>::free_bit(int64_t offset) {
  int64_t mask_offset = offset / num_bits_per_value;
  int64_t bit_offset = offset % num_bits_per_value;
  values_[mask_offset] |= 1 << bit_offset;
  next_free_bit_ = std::min(offset, next_free_bit_);
}
template <typename T>
inline bool Bitmap<T>::full() const {
  return next_free_bit_ >= num_total_bits_;
}

} // namespace torchrec
