/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/bits_op.h>

#include <bit>

namespace torchrec::bits_impl {

template <typename T>
struct CtzImpl {
  /**
   * Naive implementation for no __builtin_ctz
   */
  int operator()(T v) const {
    if (v == 0)
      return -1;
    int result = 0;
    while ((v & 1) == 0) {
      v >>= 1;
      result += 1;
    }
    return result;
  }
};

template <>
struct CtzImpl<unsigned int> {
  int operator()(unsigned int v) const {
    return std::countr_zero(v);
  }
};

template <>
struct CtzImpl<int> {
  int operator()(int v) const {
    return std::countr_zero(static_cast<unsigned int>(v));
  }
};

template <>
struct CtzImpl<unsigned long> {
  int operator()(unsigned long v) const {
    return std::countr_zero(v);
  }
};

template <>
struct CtzImpl<long> {
  int operator()(long v) const {
    return std::countr_zero(static_cast<unsigned long>(v));
  }
};

template <>
struct CtzImpl<unsigned long long> {
  int operator()(unsigned long long v) const {
    return std::countr_zero(v);
  }
};

template <>
struct CtzImpl<long long> {
  int operator()(long long v) const {
    return std::countr_zero(static_cast<unsigned long long>(v));
  }
};

template <typename T>
int Ctz<T>::operator()(T v) const {
  CtzImpl<T> ctz;
  return ctz(v);
}

template struct Ctz<int>;
template struct Ctz<unsigned int>;
template struct Ctz<long>;
template struct Ctz<unsigned long>;
template struct Ctz<long long>;
template struct Ctz<unsigned long long>;

// only for unittests
template struct Ctz<int8_t>;
template struct Ctz<uint8_t>;
} // namespace torchrec::bits_impl
