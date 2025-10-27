/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <torchrec/csrc/dynamic_embedding/details/io.h>
#include <torchrec/csrc/dynamic_embedding/details/notification.h>
#include <deque>
#include <utility>

namespace torchrec {

/**
 * @brief A local shard of embedding tensor with its range of row.
 * It not only stores the parameter tensor of the shard, but also
 * the tensor of this optimizer states.
 *
 */
struct LocalShard {
  int64_t row_start;
  int64_t row_size;
  std::vector<torch::Tensor> tensors;

  /**
   * @brief Check if a certain cache id is in this Shard
   *
   */
  [[nodiscard]] bool has(int64_t cache_id) const {
    return row_start <= cache_id && cache_id < row_start + row_size;
  }

  [[nodiscard]] std::vector<torch::Tensor> get_tensor_view(
      int64_t cache_id) const {
    std::vector<torch::Tensor> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) {
      result.emplace_back(
          tensor.slice(0, cache_id - row_start, cache_id - row_start + 1));
    }
    return result;
  }
};

/**
 * @brief A helper class to store all the local shard on the current rank,
 * basically `std::vecotr<LocalShard>`. The reason for this class is that all
 * shards could share the same refcount.
 *
 */
class LocalShardList : public torch::CustomClassHolder {
  using Container = std::vector<LocalShard>;

 public:
  void emplace_back(
      int64_t row_start,
      int64_t col_start,
      int64_t row_size,
      int64_t col_size,
      std::vector<torch::Tensor> tensors) {
    // col_start/col_size not supported now.
    shards_.emplace_back(
        LocalShard{
            .row_start = row_start,
            .row_size = row_size,
            .tensors = std::move(tensors)});
  }

  Container::const_iterator begin() const {
    return shards_.begin();
  }

  Container::const_iterator end() const {
    return shards_.end();
  }

  Container shards_;
};

class FetchHandle;

class PS : public torch::CustomClassHolder {
 public:
  PS(std::string table_name,
     c10::intrusive_ptr<LocalShardList> shards,
     int64_t col_size,
     int64_t num_optimizer_stats,
     const std::string& io_config,
     int64_t chunk_size)
      : table_name_(std::move(table_name)),
        shards_(std::move(shards)),
        col_size_(col_size),
        os_ids_(num_optimizer_stats),
        io_(io_config),
        num_ids_per_chunk_(chunk_size / col_size_ / num_optimizer_stats) {
    TORCH_CHECK(num_ids_per_chunk_ > 0, "chunk size too small");
    for (int64_t i = 0; i < num_optimizer_stats; ++i) {
      os_ids_[i] = i;
    }
  }

  /**
   * @brief Fetch the embedding from remote PS into local GPU embedding
   * asynchronously.
   *
   * @param ids_to_fetch ids to fetch, pairs of global id and cache id.
   * @param time the timestamp of the fetch
   * @param reinit whether to re-initialize the parameter and optimizer states
   * if the id to fetch is not stored in PS. The parameter will be re-initialize
   * with `uniform(weight_init_min, weight_init_max)` and the optimizer states
   * will be re-initialized with 0.
   * @return The handle used to synchronize the fetch.
   */
  c10::intrusive_ptr<FetchHandle> fetch(
      torch::Tensor ids_to_fetch,
      int64_t time,
      bool reinit,
      double weight_init_min,
      double weight_init_max);
  /**
   * @brief Synchronize all the fetches till timestamp `time`,
   * if `time` is -1, then synchronize all previous fetches.
   *
   */
  void synchronize_fetch(int64_t time = -1);

  /**
   * @brief Evict ids back to PS synchronously.
   *
   */
  void evict(torch::Tensor ids_to_evict);

 private:
  std::vector<torch::Tensor> get_tensor_views(int64_t cache_id);
  std::tuple<std::vector<int64_t>, std::vector<int64_t>> filter_local_ids(
      const torch::Tensor& ids);

  // We need a mutex because the evict and fetch may happen in different thread.
  std::mutex mu_;
  std::mutex fetch_notifications_mutex_;
  std::string table_name_;
  c10::intrusive_ptr<LocalShardList> shards_;
  int64_t col_size_;
  std::vector<uint32_t> os_ids_;
  int64_t num_ids_per_chunk_;
  IO io_;
  std::deque<std::pair<int64_t, c10::intrusive_ptr<Notification>>>
      fetch_notifications_;
};

struct FetchHandle : public torch::CustomClassHolder {
 public:
  FetchHandle(int64_t time, c10::intrusive_ptr<PS> ps)
      : time_(time), ps_(std::move(ps)) {}
  void wait() {
    if (ps_ != nullptr)
      ps_->synchronize_fetch(time_);
  }

 private:
  int64_t time_;
  c10::intrusive_ptr<PS> ps_; // not owned
};

} // namespace torchrec
