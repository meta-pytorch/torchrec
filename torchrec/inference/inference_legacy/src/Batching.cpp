/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Batching.h" // @manual

#include <c10/core/ScalarType.h>
#include <folly/Range.h>
#include <folly/container/Enumerate.h>
#include <folly/io/Cursor.h>

#include "ATen/Functions.h"
#include "ATen/core/List.h"
#include "ATen/core/ivalue.h"
#include "torchrec/inference/Types.h"

namespace torchrec {

void moveIValueToDevice(c10::IValue& val, const c10::Device& device) {
  if (val.isTensor()) {
    if (val.toTensor().device() != device) {
      val = val.toTensor().to(device, /* non_blocking */ true);
    }
  } else if (val.isList()) {
    for (auto v : val.toListRef()) {
      moveIValueToDevice(v, device);
    }
  } else {
    LOG(WARNING)
        << "moveIValueToDevice only supports types c10::List and at::Tensor but received type "
        << val.type().get()->repr_str();
  }
}

std::unordered_map<std::string, c10::IValue> moveToDevice(
    std::unordered_map<std::string, c10::IValue> combined,
    const c10::Device& device) {
  for (auto& [k, v] : combined) {
    moveIValueToDevice(v, device);
  }
  return combined;
}

C10_DEFINE_REGISTRY(TorchRecBatchingFuncRegistry, BatchingFunc);

std::unordered_map<std::string, c10::IValue> combineFloat(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests) {
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  auto* maybeIValuePtr =
      std::get_if<c10::IValue>(&requests.front()->features[featureName]);
  at::Tensor combined;
  std::vector<at::Tensor> tensors;

  if (maybeIValuePtr != nullptr) {
    numFeatures = maybeIValuePtr->toTensor().size(1);
    tensors.reserve(requests.size());
  }

  for (const auto& request : requests) {
    if (maybeIValuePtr != nullptr) {
      tensors.push_back(
          std::get<c10::IValue>(request->features[featureName]).toTensor());
      combinedBatchSize += tensors.back().size(0);
    } else {
      const auto& feature =
          std::get<torchrec::FloatFeatures>(request->features[featureName]);
      const auto nf = feature.num_features;
      const auto dataSize = feature.values.computeChainDataLength();
      if (nf * request->batch_size * sizeof(float) != dataSize) {
        throw std::invalid_argument("Invalid float features");
      }
      if (nf > 0) {
        combinedBatchSize += request->batch_size;
        if (numFeatures > 0) {
          if (numFeatures != nf) {
            throw std::invalid_argument("Different number of float features");
          }
        }
        numFeatures = nf;
      }
    }
  }

  if (numFeatures == 0) {
    return {{featureName, at::empty({combinedBatchSize, 0})}};
  }

  if (maybeIValuePtr != nullptr) {
    combined = at::empty(
        {combinedBatchSize, numFeatures},
        maybeIValuePtr->toTensor().options().pinned_memory(true));
    at::cat_out(combined, tensors);
  } else {
    // Create output tensor.
    const auto options =
        at::TensorOptions(at::kCPU).dtype(at::kFloat).pinned_memory(true);
    combined = at::empty({combinedBatchSize, numFeatures}, options);

    // Copy tensor data.
    auto combinedRange = folly::MutableByteRange(
        reinterpret_cast<uint8_t*>(combined.data_ptr()),
        combinedBatchSize * numFeatures * options.dtype().itemsize());
    for (const auto& request : requests) {
      const auto* start =
          &std::get<torchrec::FloatFeatures>(request->features[featureName])
               .values;
      const auto* curr = start;
      do {
        std::memcpy(combinedRange.data(), curr->data(), curr->length());
        combinedRange.advance(curr->length());
        curr = curr->next();
      } while (curr != start);
    }
  }

  return {{featureName, std::move(combined)}};
}

std::unordered_map<std::string, c10::IValue> combineSparse(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests,
    bool isWeighted) {
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  long totalLength = 0;
  // request -> feature -> length of the feature
  std::vector<std::vector<long>> featureLengths;
  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::SparseFeatures>(request->features[featureName]);

    // validate num_features
    const auto nf = features.num_features;
    if (nf > 0) {
      combinedBatchSize += request->batch_size;
      if (numFeatures > 0) {
        if (numFeatures != nf) {
          throw std::invalid_argument("Different number of float features");
        }
      }
      numFeatures = nf;
    }

    featureLengths.emplace_back().reserve(nf);
    size_t requestLength = 0;
    folly::io::Cursor lengthsCursor(&features.lengths);
    for (int i = 0; i < features.num_features; ++i) {
      size_t featureLength = 0;
      for (int j = 0; j < request->batch_size; ++j) {
        featureLength += lengthsCursor.read<int32_t>();
      }
      featureLengths.back().push_back(featureLength);
      requestLength += featureLength;
    }
    CHECK(lengthsCursor.isAtEnd());
    totalLength += requestLength;
  }

  // Create output tensor.
  const auto options = at::TensorOptions(at::kCPU).pinned_memory(true);
  auto lengths =
      at::empty({numFeatures * combinedBatchSize}, options.dtype(at::kInt));
  auto values = at::empty({totalLength}, options.dtype(at::kInt));
  auto weights =
      at::empty({isWeighted ? totalLength : 0}, options.dtype(at::kFloat));

  std::vector<folly::io::Cursor> lengthsCursors;
  std::vector<folly::io::Cursor> valuesCursor;
  std::vector<folly::io::Cursor> weightsCursor;
  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::SparseFeatures>(request->features[featureName]);

    lengthsCursors.emplace_back(&features.lengths);
    valuesCursor.emplace_back(&features.values);
    if (isWeighted) {
      weightsCursor.emplace_back(&features.weights);
    }
  }

  auto lengthsRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(lengths.data_ptr()),
      lengths.numel() * lengths.dtype().itemsize());
  auto valuesRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(values.data_ptr()),
      values.numel() * values.dtype().itemsize());
  auto weightsRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(weights.data_ptr()),
      isWeighted ? weights.numel() * weights.dtype().itemsize() : 0);

  for (int i = 0; i < numFeatures; ++i) {
    for (int j = 0; j < requests.size(); ++j) {
      const auto& request = requests[j];

      // TODO: determine values elem size by value_type.
      size_t len = request->batch_size * sizeof(int32_t);
      lengthsCursors[j].pull(lengthsRange.data(), len);
      lengthsRange.advance(len);

      len = featureLengths[j][i] * sizeof(int32_t);
      valuesCursor[j].pull(valuesRange.data(), len);
      valuesRange.advance(len);

      if (isWeighted) {
        len = featureLengths[j][i] * sizeof(float);
        weightsCursor[j].pull(weightsRange.data(), len);
        weightsRange.advance(len);
      }
    }
  }

  std::unordered_map<std::string, c10::IValue> ret = {
      {featureName + ".values", std::move(values)},
      {featureName + ".lengths", std::move(lengths)},
  };
  if (isWeighted) {
    ret[featureName + ".weights"] = std::move(weights);
  }
  return ret;
}

std::unordered_map<std::string, c10::IValue> combineEmbedding(
    const std::string& featureName,
    const std::vector<std::shared_ptr<PredictionRequest>>& requests) {
  // Compute combined batch size.
  long combinedBatchSize = 0;
  long numFeatures = 0;
  long dimension = 0;

  // If input is IValue then we expect a List[Tensor] of length numFeatures
  // Each element of this list is a batch of features with size (batchSize x
  // dimension)
  auto* maybeIValuePtr =
      std::get_if<c10::IValue>(&requests.front()->features[featureName]);

  for (const auto& request : requests) {
    if (maybeIValuePtr != nullptr) {
      auto ival = std::get<c10::IValue>(request->features[featureName])
                      .toTensorVector();
      auto nf = ival.size();
      if (nf == 0) {
        continue;
      }
      if (numFeatures > 0 && nf > 0 && numFeatures != nf) {
        throw std::invalid_argument("Different number of embedding features");
      }
      numFeatures = nf;
      combinedBatchSize += ival.at(0).size(0);
    } else {
      const auto& features =
          std::get<torchrec::FloatFeatures>(request->features[featureName]);
      const auto nf = features.num_features;
      const auto dataSize = features.values.computeChainDataLength();
      if (nf != 0 && dimension == 0) {
        dimension = dataSize / request->batch_size / sizeof(float) / nf;
      }
      if (nf * request->batch_size * dimension * sizeof(float) != dataSize) {
        throw std::invalid_argument("Invalid embedding features");
      }
      if (nf > 0) {
        combinedBatchSize += request->batch_size;
        if (numFeatures > 0) {
          if (numFeatures != nf) {
            throw std::invalid_argument(
                "Different number of embedding features");
          }
        }
        numFeatures = nf;
      }
    }
  }

  if (numFeatures == 0) {
    return {{featureName, c10::List<at::Tensor>()}};
  }

  std::vector<folly::io::Cursor> cursors;
  if (maybeIValuePtr != nullptr) {
    std::vector<std::vector<at::Tensor>> featureBatches(numFeatures);
    for (const auto& request : requests) {
      auto ival = std::get<c10::IValue>(request->features[featureName])
                      .toTensorVector();
      if (ival.size() == 0) {
        continue;
      }
      for (int i = 0; i < numFeatures; ++i) {
        auto featureBatch = ival.at(i);
        if (featureBatch.dim() == 1) {
          featureBatch = featureBatch.unsqueeze(1);
        }
        featureBatches.at(i).push_back(featureBatch);
      }
    }

    c10::List<at::Tensor> retList;
    for (const auto& fb : featureBatches) {
      retList.push_back(at::cat(fb));
    }
    return {{featureName, std::move(retList)}};
  }

  for (const auto& request : requests) {
    const auto& features =
        std::get<torchrec::FloatFeatures>(request->features[featureName]);
    cursors.emplace_back(&features.values);
  }

  // Create output tensor.
  const auto options =
      at::TensorOptions(at::kCPU).dtype(at::kFloat).pinned_memory(true);
  auto combined =
      at::empty({combinedBatchSize, numFeatures, dimension}, options);

  // Copy tensor data.
  auto combinedRange = folly::MutableByteRange(
      reinterpret_cast<uint8_t*>(combined.data_ptr()),
      combined.storage().nbytes());

  for (const auto&& it : folly::enumerate(cursors)) {
    auto len = requests[it.index]->batch_size * dimension * numFeatures *
        sizeof(float);
    it.element.pull(combinedRange.data(), len);
    combinedRange.advance(len);
  }

  auto listFeatureBatches = c10::List<at::Tensor>();
  for (auto& tensor : combined.transpose(0, 1).split(1)) {
    listFeatureBatches.push_back(tensor.squeeze(0));
  }
  return {{featureName, std::move(listFeatureBatches)}};
}

namespace {

class FloatBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, c10::IValue> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& device,
      LazyTensorRef /* batchItems */) override {
    return moveToDevice(combineFloat(featureName, requests), device);
  }
};

class SparseBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, c10::IValue> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& device,
      LazyTensorRef /* batchItems */) override {
    return moveToDevice(
        combineSparse(featureName, requests, /* isWeighted */ false), device);
  }
};

class WeightedSparseBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, c10::IValue> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& device,
      LazyTensorRef /* batchItems */) override {
    return moveToDevice(
        combineSparse(featureName, requests, /* isWeighted */ true), device);
  }
};

class EmbeddingBatchingFunc : public BatchingFunc {
 public:
  std::unordered_map<std::string, c10::IValue> batch(
      const std::string& featureName,
      const std::vector<std::shared_ptr<PredictionRequest>>& requests,
      const int64_t& /* totalNumBatch */,
      LazyTensorRef /* batchOffsets */,
      const c10::Device& device,
      LazyTensorRef /* batchItems */) override {
    return moveToDevice(combineEmbedding(featureName, requests), device);
  }
};

} // anonymous namespace

REGISTER_TORCHREC_BATCHING_FUNC(dense, FloatBatchingFunc);
REGISTER_TORCHREC_BATCHING_FUNC(sparse, SparseBatchingFunc);
REGISTER_TORCHREC_BATCHING_FUNC(weighted_sparse, WeightedSparseBatchingFunc);
REGISTER_TORCHREC_BATCHING_FUNC(embedding, EmbeddingBatchingFunc);

} // namespace torchrec
