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

#include <deque>
#include <utility>
#include "tde/details/io.h"
#include "tde/notification.h"
#include "tde/tensor_list.h"

namespace tde {

struct LocalShard {
  int64_t row_start_;
  int64_t row_size_;
  c10::intrusive_ptr<TensorList> tensors_;

  [[nodiscard]] bool Has(int64_t cache_id) const {
    return row_start_ <= cache_id && cache_id < row_start_ + row_size_;
  }

  [[nodiscard]] std::vector<torch::Tensor> GetTensorView(
      int64_t cache_id) const {
    std::vector<torch::Tensor> result;
    result.reserve(tensors_->size());
    for (auto& tensor : *tensors_) {
      result.emplace_back(
          tensor.slice(0, cache_id - row_start_, cache_id - row_start_ + 1));
    }
    return result;
  }
};

class LocalShardList : public torch::CustomClassHolder {
  using Container = std::vector<LocalShard>;

 public:
  void emplace_back(
      int64_t row_start,
      int64_t col_start,
      int64_t row_size,
      int64_t col_size,
      c10::intrusive_ptr<TensorList> tensors) {
    // col_start/col_size not used for now.
    shards_.emplace_back(
        LocalShard{
            .row_start_ = row_start,
            .row_size_ = row_size,
            .tensors_ = std::move(tensors)});
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

  c10::intrusive_ptr<FetchHandle> Fetch(
      torch::Tensor ids_to_fetch,
      int64_t time,
      bool reinit,
      double weight_init_min,
      double weight_init_max);
  void Evict(torch::Tensor ids_to_evict);

  void SyncFetch(int64_t time = -1);

 private:
  std::vector<torch::Tensor> GetTensorViews(int64_t cache_id);
  std::vector<int64_t> global_ids_to_fetch_or_evict_;
  std::vector<int64_t> cache_ids_to_fetch_or_evict_;

  void Filter(const torch::Tensor& tensor);

  std::mutex mu_;
  std::mutex fetch_notifications_mutex_;
  std::string table_name_;
  c10::intrusive_ptr<LocalShardList> shards_;
  int64_t col_size_;
  std::vector<uint32_t> os_ids_;
  int64_t num_ids_per_chunk_;
  details::IO io_;
  std::deque<std::pair<int64_t, c10::intrusive_ptr<Notification>>>
      fetch_notifications_;
};

struct FetchHandle : public torch::CustomClassHolder {
 public:
  FetchHandle(int64_t time, c10::intrusive_ptr<PS> ps)
      : time_(time), ps_(std::move(ps)) {}
  void Wait() {
    if (ps_ != nullptr)
      ps_->SyncFetch(time_);
  }

 private:
  int64_t time_;
  c10::intrusive_ptr<PS> ps_; // not owned
};

} // namespace tde
