/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <limits>
#include "gtest/gtest.h"
#include "tde/details/cacheline_id_transformer.h"

namespace tde::details {

TEST(tde, CachelineThreadedIDTransformer_NoFilter) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 0, 2, 1};
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));
  for (size_t i = 0; i < 5; i++) {
    ASSERT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, CachelineThreadedIDTransformer_Full) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(4);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 2, 3, -1};
  EXPECT_FALSE(transformer.Transform(global_ids, cache_ids));
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, CachelineThreadedIDTransformer_Evict) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(4);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];

  EXPECT_FALSE(transformer.Transform(global_ids, cache_ids));

  const int64_t evict_global_ids[2] = {100, 102};
  transformer.Evict(evict_global_ids);

  const int64_t new_global_ids[4] = {101, 102, 103, 104};
  int64_t new_cache_ids[4];

  EXPECT_TRUE(transformer.Transform(new_global_ids, new_cache_ids));

  int64_t expected_cache_ids[4] = {1, 0, 3, 2};
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(expected_cache_ids[i], new_cache_ids[i]);
  }
}

TEST(tde, CachelineThreadedIDTransformer_Iterator) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 0, 2, 0};
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));

  auto iterator = transformer.Iterator();
  for (size_t i = 0; i < 3; i++) {
    ASSERT_TRUE(iterator().has_value());
  }
  ASSERT_TRUE(!iterator().has_value());
}

TEST(tde, CachelineIDTransformer_RejectsNegativeIDs) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[2] = {100, -1};
  int64_t cache_ids[2];
  EXPECT_THROW(transformer.Transform(global_ids, cache_ids), c10::Error);

  const int64_t min_id[1] = {std::numeric_limits<int64_t>::min()};
  int64_t min_cache_id[1];
  EXPECT_THROW(transformer.Transform(min_id, min_cache_id), c10::Error);

  // 100 precedes -1 in the batch but must not have been inserted.
  EXPECT_FALSE(transformer.Iterator()().has_value());
}

TEST(tde, CachelineIDTransformer_EvictIgnoresNegativeIDs) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[2] = {100, 101};
  int64_t cache_ids[2];
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));

  const int64_t evict_ids[2] = {-1, std::numeric_limits<int64_t>::min()};
  transformer.Evict(evict_ids);

  int64_t again[2];
  ASSERT_TRUE(transformer.Transform(global_ids, again));
  EXPECT_EQ(cache_ids[0], again[0]);
  EXPECT_EQ(cache_ids[1], again[1]);

  // Cache ids 0 and 1 must still be held, so the next id gets 2.
  const int64_t new_id[1] = {102};
  int64_t new_cache_id[1];
  ASSERT_TRUE(transformer.Transform(new_id, new_cache_id));
  EXPECT_EQ(2, new_cache_id[0]);
}

} // namespace tde::details
