/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <vector>
#include "tde/details/bits_op.h"

namespace tde::details {

static void* alignMalloc(size_t align, size_t size) {
  void* result;
  TORCH_CHECK(
      posix_memalign(&result, align, size) == 0,
      "posix_memalign error, ",
      errno,
      ": ",
      strerror(errno));
  return result;
}

template <
    typename LXURecord,
    int64_t NumCacheline,
    int64_t CachelineSize,
    typename BitMap,
    typename Hash>
inline CachelineIDTransformer<
    LXURecord,
    NumCacheline,
    CachelineSize,
    BitMap,
    Hash>::CachelineIDTransformer(int64_t num_embedding, int64_t capacity)
    : num_groups_(
          ((capacity == 0 ? 2 * num_embedding : capacity) + group_size_ - 1) /
          group_size_) /*capacity by default is 2 * num_embedding */,
      cache_values_(
          reinterpret_cast<CacheValue*>(alignMalloc(
              CachelineSize,
              sizeof(CacheValue) * num_groups_ * group_size_))),
      bitmap_(num_embedding) {
  memset(
      cache_values_.get(), 0, sizeof(CacheValue) * num_groups_ * group_size_);
}

template <
    typename LXURecord,
    int64_t NumCacheline,
    int64_t CachelineSize,
    typename BitMap,
    typename Hash>
template <typename Update, typename Fetch>
inline bool CachelineIDTransformer<
    LXURecord,
    NumCacheline,
    CachelineSize,
    BitMap,
    Hash>::
    Transform(
        tcb::span<const int64_t> global_ids,
        tcb::span<int64_t> cache_ids,
        Update update,
        Fetch fetch) {
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    auto [group_id, intra_id] = FindGroupIndex(global_id);
    int64_t global_id_not = ~global_id;
    CacheValue* group_begin = &cache_values_[group_id * group_size_];
    int64_t k = 0;
    int64_t empty_slot = -1;
    int64_t cache_id = -1;
    for (; k < group_size_; k++, intra_id++) {
      intra_id %= group_size_;
      auto& cache_value = group_begin[intra_id];
      // tricky but fast :p
      int64_t xor_value = cache_value.global_id_not_ ^ global_id_not;

      if (xor_value == 0) { // found
        cache_id = cache_value.cache_id_;
        cache_value.lxu_record_ =
            update(cache_value.lxu_record_, global_id, cache_id);
        break;
      } else if (xor_value < 0 && empty_slot < 0) { // empty slot
        empty_slot = intra_id;
      }
    }
    if (cache_id < 0) {
      if (empty_slot >= 0) {
        // The transformer is full.
        if (C10_UNLIKELY(bitmap_.Full())) {
          return false;
        }
        auto& cache_value = group_begin[empty_slot];
        cache_id = bitmap_.NextFreeBit();
        cache_value.global_id_not_ = global_id_not;
        cache_value.cache_id_ = cache_id;
        cache_value.lxu_record_ = update(std::nullopt, global_id, cache_id);
        fetch(global_id, cache_id);
      } else {
        return false;
      }
    }
    cache_ids[i] = cache_id;
  }
  return true;
}

template <
    typename LXURecord,
    int64_t NumCacheline,
    int64_t CachelineSize,
    typename BitMap,
    typename Hash>
inline void CachelineIDTransformer<
    LXURecord,
    NumCacheline,
    CachelineSize,
    BitMap,
    Hash>::Evict(tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto [group_id, intra_id] = FindGroupIndex(global_id);

    int64_t global_id_not = ~global_id;
    for (int64_t k = 0; k < group_size_; k++) {
      int64_t offset = group_id * group_size_ + (intra_id + k) % group_size_;
      auto& cache_value = cache_values_[offset];
      // tricky but fast :p
      int64_t xor_value = global_id_not ^ cache_value.global_id_not_;
      if (xor_value < 0) { // not exist
        continue;
      } else if (xor_value == 0) { // found slot
        bitmap_.FreeBit(cache_value.cache_id_);
        cache_value.global_id_not_ = 0;
        break;
      }
    }
  }
}

} // namespace tde::details
