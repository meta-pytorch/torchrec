/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace torchrec::torch_tpu {
namespace {

namespace py = ::pybind11;

class SparseCorePreprocessorBackend {
 public:
  SparseCorePreprocessorBackend(const py::list& tables_metadata,
                                int64_t batch_size, int64_t local_device_count,
                                int64_t global_device_count,
                                int64_t num_sc_per_device,
                                bool allow_id_dropping)
      : local_device_count_(local_device_count),
        global_device_count_(global_device_count),
        jax_options_{
            .local_device_count = static_cast<int>(local_device_count),
            .global_device_count = static_cast<int>(global_device_count),
            .num_sc_per_device = static_cast<int>(num_sc_per_device),
            .sharding_strategy = jax_sc_embedding::ShardingStrategy::kMod,
            .allow_id_dropping = allow_id_dropping,
        },
        row_pointers_size_per_device_(
            jax_options_.GetRowPointersSizePerDevice()) {
    int global_feat_idx = 0;
    for (const py::handle& table_obj : tables_metadata) {
      py::dict table_dict = py::cast<py::dict>(table_obj);
      std::string table_name = py::cast<std::string>(table_dict["name"]);
      table_names_.push_back(table_name);
      int max_ids = py::cast<int>(table_dict["max_ids_per_partition"]);
      int max_unique_ids =
          py::cast<int>(table_dict["max_unique_ids_per_partition"]);
      int buffer_size =
          py::cast<int>(table_dict["suggested_coo_buffer_size_per_device"]);

      py::list features = py::cast<py::list>(table_dict["features"]);
      jax_stacked_tables_[table_name].reserve(features.size());
      for (const py::handle& feat_obj : features) {
        py::dict feat_dict = py::cast<py::dict>(feat_obj);
        std::string feat_name = py::cast<std::string>(feat_dict["name"]);
        int row_offset = py::cast<int>(feat_dict["row_offset"]);
        int col_offset = py::cast<int>(feat_dict["col_offset"]);
        int col_shift = py::cast<int>(feat_dict["col_shift"]);
        int feat_batch_size = py::cast<int>(feat_dict["batch_size"]);
        std::string combiner_str = py::cast<std::string>(feat_dict["combiner"]);
        int max_col_id = feat_dict.contains("max_col_id")
                             ? py::cast<int>(feat_dict["max_col_id"])
                             : std::numeric_limits<int>::max();

        jax_stacked_tables_[table_name].push_back(
            jax_sc_embedding::FeatureMetadataInStack(
                feat_name, global_feat_idx++, max_ids, max_unique_ids,
                row_offset, col_offset, col_shift, feat_batch_size,
                buffer_size > 0 ? std::make_optional(buffer_size)
                                : std::nullopt,
                jax_sc_embedding::GetRowCombiner(combiner_str), max_col_id));
      }
      required_buffer_sizes_[table_name] =
          jax_sc_embedding::ComputeCooBufferSizePerDevice(
              jax_options_, jax_stacked_tables_.at(table_name));
    }
  }

  std::map<std::string, std::map<std::string, at::Tensor>> Preprocess(
      const std::map<std::string, std::vector<at::Tensor>>& input_indices,
      const std::map<std::string, std::vector<at::Tensor>>& input_offsets) {
    std::vector<std::unique_ptr<jax_sc_embedding::AbstractInputBatch>>
        input_batches;
    std::vector<at::Tensor> contiguous_tensors_holder;

    int total_features = 0;
    for (const auto& [table_name, meta_list] : jax_stacked_tables_) {
      total_features += meta_list.size();
    }
    contiguous_tensors_holder.reserve(total_features * 2);
    input_batches.reserve(total_features);

    // Loop 1 (Pre-execution Input Processing & Validation):
    for (const std::string& table_name : table_names_) {
      const auto& meta_list = jax_stacked_tables_.at(table_name);

      auto indices_it = input_indices.find(table_name);
      ABSL_CHECK(indices_it != input_indices.end())
          << "input_indices missing table " << table_name;

      auto offsets_it = input_offsets.find(table_name);
      ABSL_CHECK(offsets_it != input_offsets.end())
          << "input_offsets missing table " << table_name;

      const auto& indices_list = indices_it->second;
      const auto& offsets_list = offsets_it->second;

      ABSL_CHECK_EQ(indices_list.size(), meta_list.size())
          << "indices list size mismatch for table " << table_name;
      ABSL_CHECK_EQ(offsets_list.size(), meta_list.size())
          << "offsets list size mismatch for table " << table_name;

      for (size_t i = 0; i < meta_list.size(); ++i) {
        const at::Tensor& indices = indices_list[i];
        const at::Tensor& offsets = offsets_list[i];

        ABSL_CHECK_EQ(indices.dtype(), at::kInt) << "indices must be int32";
        ABSL_CHECK_EQ(offsets.dtype(), at::kInt) << "offsets must be int32";

        auto contiguous_indices = indices.contiguous();
        auto contiguous_offsets = offsets.contiguous();
        contiguous_tensors_holder.push_back(contiguous_indices);
        contiguous_tensors_holder.push_back(contiguous_offsets);

        absl::Span<const int32_t> val_span(
            contiguous_indices.data_ptr<int32_t>(), contiguous_indices.numel());
        absl::Span<const int32_t> off_span(
            contiguous_offsets.data_ptr<int32_t>(), contiguous_offsets.numel());

        input_batches.push_back(
            std::make_unique<jax_sc_embedding::RaggedTensorInputBatch<
                absl::Span<const int32_t>, absl::Span<const int32_t>>>(
                val_span, off_span, table_name));
      }
    }

    std::map<std::string, std::map<std::string, at::Tensor>> outputs;
    jax_sc_embedding::OutputCsrArrays jax_output_buffers;

    // Loop 2 (Pre-execution Output Allocation & Buffer Setup):
    for (const std::string& table_name : table_names_) {
      const auto& meta_list = jax_stacked_tables_.at(table_name);
      if (meta_list.empty()) continue;

      int32_t required_buffer_size = required_buffer_sizes_.at(table_name);

      int total_row_pointers_size =
          local_device_count_ * row_pointers_size_per_device_;
      int total_coo_buffer_size = local_device_count_ * required_buffer_size;

      at::Tensor row_pointers = at::zeros({total_row_pointers_size}, at::kInt);
      at::Tensor embedding_ids = at::empty({total_coo_buffer_size}, at::kInt);
      at::Tensor sample_ids = at::empty({total_coo_buffer_size}, at::kInt);
      at::Tensor gains = at::empty({total_coo_buffer_size}, at::kFloat);

      Eigen::Map<jax_sc_embedding::MatrixXi> row_pointers_map(
          row_pointers.data_ptr<int32_t>(), local_device_count_,
          row_pointers_size_per_device_);
      Eigen::Map<jax_sc_embedding::MatrixXi> embedding_ids_map(
          embedding_ids.data_ptr<int32_t>(), local_device_count_,
          required_buffer_size);
      Eigen::Map<jax_sc_embedding::MatrixXi> sample_ids_map(
          sample_ids.data_ptr<int32_t>(), local_device_count_,
          required_buffer_size);
      Eigen::Map<jax_sc_embedding::MatrixXf> gains_map(
          gains.data_ptr<float>(), local_device_count_, required_buffer_size);

      jax_output_buffers.lhs_row_pointers.emplace(table_name, row_pointers_map);
      jax_output_buffers.lhs_embedding_ids.emplace(table_name,
                                                   embedding_ids_map);
      jax_output_buffers.lhs_sample_ids.emplace(table_name, sample_ids_map);
      jax_output_buffers.lhs_gains.emplace(table_name, gains_map);

      std::map<std::string, at::Tensor> table_dict;
      table_dict["row_pointers"] = row_pointers;
      table_dict["embedding_ids"] = embedding_ids;
      table_dict["sample_ids"] = sample_ids;
      table_dict["gains"] = gains;

      outputs[table_name] = table_dict;
    }

    // Execute C++ Preprocess
    auto result = jax_sc_embedding::PreprocessSparseDenseMatmulInput(
        absl::MakeSpan(input_batches), jax_stacked_tables_, jax_options_,
        &jax_output_buffers);
    ABSL_CHECK(result.ok()) << result.status().message();

    // Loop 3 (Post-execution): Extract stats for each table
    const auto& stats = result.value().stats;
    for (const std::string& table_name : table_names_) {
      auto table_it = outputs.find(table_name);
      if (table_it == outputs.end()) continue;

      int dropped_count = 0;
      auto dropped_it = stats.dropped_id_count.find(table_name);
      if (dropped_it != stats.dropped_id_count.end()) {
        dropped_count = dropped_it->second;
      }

      int observed_max_ids = 0;
      auto max_ids_it = stats.max_ids_per_partition.find(table_name);
      if (max_ids_it != stats.max_ids_per_partition.end()) {
        observed_max_ids = static_cast<int>(max_ids_it->second.maxCoeff());
      }

      int observed_max_unique_ids = 0;
      auto max_unique_it = stats.max_unique_ids_per_partition.find(table_name);
      if (max_unique_it != stats.max_unique_ids_per_partition.end()) {
        observed_max_unique_ids =
            static_cast<int>(max_unique_it->second.maxCoeff());
      }

      table_it->second["dropped_count"] =
          at::scalar_tensor(dropped_count, at::kInt);
      table_it->second["observed_max_ids"] =
          at::scalar_tensor(observed_max_ids, at::kInt);
      table_it->second["observed_max_unique_ids"] =
          at::scalar_tensor(observed_max_unique_ids, at::kInt);
    }

    return outputs;
  }

 private:
  int64_t local_device_count_;
  int64_t global_device_count_;
  jax_sc_embedding::PreprocessSparseDenseMatmulInputOptions jax_options_;
  int row_pointers_size_per_device_;
  absl::flat_hash_map<std::string,
                      std::vector<jax_sc_embedding::FeatureMetadataInStack>>
      jax_stacked_tables_;
  absl::flat_hash_map<std::string, int32_t> required_buffer_sizes_;
  std::vector<std::string> table_names_;
};

}  // namespace

PYBIND11_MODULE(pybind_input_preprocessing, m) {
  py::module_::import("torch");
  py::class_<SparseCorePreprocessorBackend>(
      m, "SparseCorePreprocessorBackend",
      "Stateful C++ backend for CPU dataset preprocessing of SparseCore "
      "embedding tables.")
      .def(py::init<py::list, int64_t, int64_t, int64_t, int64_t, bool>(),
           py::arg("tables_metadata"), py::arg("batch_size"),
           py::arg("local_device_count"), py::arg("global_device_count"),
           py::arg("num_sc_per_device"), py::arg("allow_id_dropping") = false,
           "Initializes the preprocessor backend, pre-computing stacked table "
           "metadata and required buffer sizes.")
      .def("preprocess", &SparseCorePreprocessorBackend::Preprocess,
           py::arg("input_indices"), py::arg("input_offsets"),
           "Preprocesses per-batch input indices and offsets into CSR-wrapped "
           "COO format.");
}

}  // namespace torchrec::torch_tpu
