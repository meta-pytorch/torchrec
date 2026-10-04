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
int Ctz<T>::operator()(T v) const {
  using UT = std::make_unsigned_t<T>;
  return std::countr_zero(static_cast<UT>(v));
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
