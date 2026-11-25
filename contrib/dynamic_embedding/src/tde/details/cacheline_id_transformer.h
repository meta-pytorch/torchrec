/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>
#include <optional>
#include "nlohmann/json.hpp"
#include "tcb/span.hpp"
#include "tde/details/move_only_function.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

template <typename LXURecord>
struct CachelineIDTransformerValue {
  int64_t global_id_not_;
  uint32_t cache_id_;
  LXURecord lxu_record_;
};

template <typename LXURecord>
class CachelineIDTransformerIterator {
  using Val = CachelineIDTransformerValue<LXURecord>;

 public:
  CachelineIDTransformerIterator(Val* begin, Val* end)
      : begin_(begin), end_(end) {}

  std::optional<TransformerRecord<LXURecord>> operator()() {
    for (; begin_ != end_;) {
      auto& record = *begin_++;
      if (record.global_id_not_ >= 0) {
        continue;
      }
      TransformerRecord<LXURecord> result{};
      result.global_id_ = ~record.global_id_not_;
      result.cache_id_ = record.cache_id_;
      result.lxu_record_ = record.lxu_record_;
      return result;
    }
    return std::nullopt;
  }

 private:
  Val* begin_;
  Val* end_;
};

/**
 * CachelineIDTransformer
 *
 * Transform GlobalID to CacheID by naive flat hash map
 * @tparam LXURecord The extension type used for eviction strategy.
 * @tparam Bitmap The bitmap class to record the free cache ids.
 */
template <
    typename LXURecord,
    int64_t NumCacheline = 8,
    int64_t CachelineSize = 64,
    typename BitMap = Bitmap<uint32_t>,
    typename Hash = std::hash<int64_t>>
class CachelineIDTransformer {
 public:
  static_assert(NumCacheline > 0, "NumCacheline should be positive.");
  static_assert(CachelineSize > 0, "CachelineSize should be positive.");
  using Self = CachelineIDTransformer<
      LXURecord,
      NumCacheline,
      CachelineSize,
      BitMap,
      Hash>;
  static constexpr std::string_view type_ = "cacheline";

  explicit CachelineIDTransformer(int64_t num_embedding, int64_t capacity = 0);

  CachelineIDTransformer(const Self&) = delete;
  CachelineIDTransformer(Self&&) noexcept = default;

  static Self Create(int64_t num_embedding, const nlohmann::json& json) {
    return Self(num_embedding);
  }

  /**
   * Transform global ids to cache ids
   *
   * @tparam Update Update the eviction strategy tag type. Update LXU Record
   * @tparam Fetch Fetch the not existing global-id/cache-id pair. It is used
   * by dynamic embedding parameter server.
   *
   * @param global_ids Global ID vector
   * @param cache_ids [out] Cache ID vector
   * @param update update lambda. See `Update` doc.
   * @param fetch fetch lambda. See `Fetch` doc.
   * @return true if all transformed, otherwise need eviction.
   * @throws c10::Error if any global id is negative.
   */
  template <
      typename Update = decltype(transform_default::NoUpdate<LXURecord>),
      typename Fetch = decltype(transform_default::NoFetch)>
  bool Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Update update = transform_default::NoUpdate<LXURecord>,
      Fetch fetch = transform_default::NoFetch);

  void Evict(tcb::span<const int64_t> global_ids);

  CachelineIDTransformerIterator<LXURecord> Iterator() const {
    return CachelineIDTransformerIterator<LXURecord>(
        cache_values_.get(), cache_values_.get() + num_groups_ * group_size_);
  }

 private:
  using CacheValue = CachelineIDTransformerValue<LXURecord>;
  static_assert(sizeof(CacheValue) <= 16);
  static_assert(std::is_trivially_destructible_v<CacheValue>);
  static_assert(std::is_trivially_constructible_v<CacheValue>);
  [[nodiscard]] std::tuple<int64_t, int64_t> FindGroupIndex(int64_t val) const {
    uint64_t hash = hasher_(val);
    return {
        static_cast<int64_t>(hash / group_size_ % num_groups_),
        static_cast<int64_t>(hash % group_size_)};
  }

  static constexpr int64_t group_size_ =
      NumCacheline * CachelineSize / static_cast<int64_t>(sizeof(CacheValue));
  int64_t num_groups_;
  Hash hasher_;

  struct CacheValueDeleter {
    void operator()(void* ptr) const {
      free(ptr);
    }
  };

  std::unique_ptr<CacheValue[], CacheValueDeleter> cache_values_;
  BitMap bitmap_;
};

} // namespace tde::details

#include "tde/details/cacheline_id_transformer_impl.h"
