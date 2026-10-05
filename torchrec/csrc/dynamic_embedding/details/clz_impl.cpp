/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/bits_op.h>

#include <bit>
#include <type_traits>

namespace torchrec::bits_impl {

template <typename T>
int Clz<T>::operator()(T v) const {
  using UT = std::make_unsigned_t<T>;
  return std::countl_zero(static_cast<UT>(v));
}

template struct Clz<int>;
template struct Clz<unsigned int>;
template struct Clz<long>;
template struct Clz<unsigned long>;
template struct Clz<long long>;
template struct Clz<unsigned long long>;

// only for unittests
template struct Clz<int8_t>;
template struct Clz<uint8_t>;
} // namespace torchrec::bits_impl
