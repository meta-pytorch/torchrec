#pragma once
#include <algorithm>
#include <vector>
#include "tde/details/bits_op.h"

namespace tde::details {

template <typename T>
inline Bitmap<T>::Bitmap(int64_t num_bits)
    : num_total_bits_(num_bits),
      num_values_((num_bits + num_bits_per_value - 1) / num_bits_per_value),
      values_(new T[num_values_]),
      next_free_bit_(0) {
  std::fill(values_.get(), values_.get() + num_values_, -1);
}

template <typename T>
inline int64_t Bitmap<T>::NextFreeBit() {
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
    next_free_bit_ = offset * num_bits_per_value + Ctz(value);
  } else {
    next_free_bit_ = num_total_bits_;
  }

  return result;
}

template <typename T>
inline void Bitmap<T>::FreeBit(int64_t offset) {
  int64_t mask_offset = offset / num_bits_per_value;
  int64_t bit_offset = offset % num_bits_per_value;
  values_[mask_offset] |= 1 << bit_offset;
  next_free_bit_ = std::min(offset, next_free_bit_);
}
template <typename T>
inline bool Bitmap<T>::Full() const {
  return next_free_bit_ >= num_total_bits_;
}

template <typename LXURecord, typename T>
inline NaiveIDTransformer<LXURecord, T>::NaiveIDTransformer(
    int64_t num_embedding)
    : bitmap_(num_embedding) {
  global_id2cache_value_.reserve(num_embedding);
}

template <typename LXURecord, typename T>
template <typename Update, typename Fetch>
inline bool NaiveIDTransformer<LXURecord, T>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Update update,
    Fetch fetch) {
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    auto iter = global_id2cache_value_.find(global_id);
    // cache_id is in [0, num_embedding)
    int64_t cache_id;
    if (iter != global_id2cache_value_.end()) {
      cache_id = iter->second.cache_id_;
      iter->second.lxu_record_ =
          update(iter->second.lxu_record_, global_id, cache_id);
    } else {
      // The transformer is full.
      if (bitmap_.Full()) [[unlikely]] {
        return false;
      }
      auto stored_cache_id = bitmap_.NextFreeBit();
      cache_id = stored_cache_id;
      LXURecord record = update(std::nullopt, global_id, cache_id);
      global_id2cache_value_.emplace(
          global_id, CacheValue{stored_cache_id, record});
      fetch(global_id, cache_id);
    }
    cache_ids[i] = cache_id;
  }
  return true;
}

template <typename LXURecord, typename T>
inline void NaiveIDTransformer<LXURecord, T>::Evict(
    tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto iter = global_id2cache_value_.find(global_id);
    if (iter == global_id2cache_value_.end()) {
      continue;
    }
    int64_t cache_id = iter->second.cache_id_;
    global_id2cache_value_.erase(iter);
    bitmap_.FreeBit(cache_id);
  }
}

template <typename LXURecord, typename T>
inline auto NaiveIDTransformer<LXURecord, T>::Iterator() const
    -> MoveOnlyFunction<std::optional<record_t>()> {
  auto iter = global_id2cache_value_.begin();
  return [iter, this]() mutable -> std::optional<record_t> {
    if (iter != global_id2cache_value_.end()) {
      auto record = record_t{
          .global_id_ = iter->first,
          .cache_id_ = iter->second.cache_id_,
          .lxu_record_ = iter->second.lxu_record_,
      };
      iter++;
      return record;
    } else {
      return {};
    }
  };
}

} // namespace tde::details
