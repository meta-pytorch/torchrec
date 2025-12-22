/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/ResultSplit.h"

#include <folly/Range.h>
#include <folly/container/Enumerate.h>
#include <folly/io/Cursor.h>

#include "torchrec/inference/Types.h"

namespace torchrec {

C10_DEFINE_REGISTRY(TorchRecResultSplitFuncRegistry, ResultSplitFunc);

c10::IValue splitDictOfTensor(
    c10::IValue result,
    size_t nOffset,
    size_t nLength,
    size_t nTotalLength) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict pred(c10::StringType::get(), c10::TensorType::get());
  pred.reserve(dict.size());

  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTensor());
    const auto& tensor = value.toTensor();
    TORCH_CHECK(tensor.dim() == 1);
    TORCH_CHECK(tensor.size(0) % nTotalLength == 0);
    const auto elemSize = tensor.size(0) / nTotalLength;
    pred.insert(
        key,
        tensor.slice(
            0, nOffset * elemSize, nOffset * elemSize + nLength * elemSize));
  }
  return pred;
}

c10::IValue splitDictOfTensors(
    c10::IValue result,
    size_t nOffset,
    size_t nLength,
    size_t nTotalLength) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict pred(
      c10::StringType::get(), c10::TupleType::create({c10::TensorType::get()}));
  pred.reserve(dict.size());

  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTuple());
    const auto tuple = value.toTuple();
    std::vector<c10::IValue> values;
    values.reserve(tuple->size());
    for (int i = 0; i < tuple->size(); ++i) {
      const auto& tensor = tuple->elements()[i].toTensor();
      TORCH_CHECK(tensor.dim() == 1);
      TORCH_CHECK(tensor.size(0) % nTotalLength == 0);
      const auto elemSize = tensor.size(0) / nTotalLength;
      values.emplace_back(tensor.slice(
          0, nOffset * elemSize, nOffset * elemSize + nLength * elemSize));
    }
    pred.insert(key, c10::ivalue::Tuple::create(std::move(values)));
  }
  return pred;
}

c10::IValue
splitDictWithMaskTensor(c10::IValue result, size_t offset, size_t length) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict pred(
      c10::StringType::get(),
      c10::TupleType::create({c10::TensorType::get(), c10::TensorType::get()}));
  pred.reserve(dict.size());
  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTuple());
    const auto& tensorTuple = value.toTuple();
    TORCH_CHECK(tensorTuple->elements().size() == 2);
    const auto& valueTensor =
        tensorTuple->elements()[0].toTensor().slice(0, offset, offset + length);
    const auto& maskTensor = tensorTuple->elements()[1].toTensor();
    auto tupleResult = c10::ivalue::Tuple::create(valueTensor, maskTensor);
    pred.insert(key, tupleResult);
  }
  return pred;
}

namespace {

class DictOfTensorResultSplitFunc : public ResultSplitFunc {
 public:
  std::string name() override {
    return "dict_of_tensor";
  }

  c10::IValue splitResult(
      c10::IValue result,
      size_t nOffset,
      size_t nLength,
      size_t nTotalLength) override {
    return splitDictOfTensor(result, nOffset, nLength, nTotalLength);
  }

  c10::IValue moveToHost(c10::IValue result) {
    const auto& dict = result.toGenericDict();
    c10::impl::GenericDict moved(
        c10::StringType::get(), c10::TensorType::get());
    moved.reserve(dict.size());

    for (auto& entry : dict) {
      const auto& key = entry.key();
      const auto& value = entry.value();
      TORCH_CHECK(value.isTensor());
      const auto& tensor = value.toTensor();
      moved.insert(
          key, tensor.to(at::Device(at::kCPU), /* non_blocking */ true));
    }
    return moved;
  }
};

class DictOfTensorsResultSplitFunc : public ResultSplitFunc {
 public:
  std::string name() override {
    return "dict_of_tensors";
  }

  c10::IValue splitResult(
      c10::IValue result,
      size_t nOffset,
      size_t nLength,
      size_t nTotalLength) override {
    return splitDictOfTensors(result, nOffset, nLength, nTotalLength);
  }

  c10::IValue moveToHost(c10::IValue result) {
    const auto& dict = result.toGenericDict();
    c10::impl::GenericDict moved(
        c10::StringType::get(),
        c10::TupleType::create({c10::TensorType::get()}));
    moved.reserve(dict.size());

    for (auto& entry : dict) {
      const auto& key = entry.key();
      const auto& value = entry.value();
      TORCH_CHECK(value.isTuple());
      const auto tuple = value.toTuple();
      std::vector<c10::IValue> values;
      values.reserve(tuple->size());
      for (int i = 0; i < tuple->size(); ++i) {
        auto& elem = tuple->elements()[i];
        TORCH_CHECK(elem.isTensor());
        const auto& tensor = elem.toTensor();
        values.emplace_back(
            tensor.to(at::Device(at::kCPU), /* non_blocking */ true));
      }
      moved.insert(key, c10::ivalue::Tuple::create(std::move(values)));
    }
    return moved;
  }
};

REGISTER_TORCHREC_RESULTSPLIT_FUNC(dict_of_tensor, DictOfTensorResultSplitFunc);

REGISTER_TORCHREC_RESULTSPLIT_FUNC(
    dict_of_tensors,
    DictOfTensorsResultSplitFunc);

REGISTER_TORCHREC_RESULTSPLIT_FUNC(
    dict_with_mask_tensor,
    DictWithMaskTensorResultSplitFunc);

} // namespace

std::string DictWithMaskTensorResultSplitFunc::name() {
  return "dict_with_mask_tensor";
}

c10::IValue DictWithMaskTensorResultSplitFunc::splitResult(
    c10::IValue result,
    size_t offset,
    size_t length,
    size_t /* nTotalLength */) {
  return splitDictWithMaskTensor(result, offset, length);
}

c10::IValue DictWithMaskTensorResultSplitFunc::moveToHost(c10::IValue result) {
  const auto& dict = result.toGenericDict();
  c10::impl::GenericDict moved(
      c10::StringType::get(),
      c10::TupleType::create({c10::TensorType::get(), c10::TensorType::get()}));
  moved.reserve(dict.size());

  for (auto& entry : dict) {
    const auto& key = entry.key();
    const auto& value = entry.value();
    TORCH_CHECK(value.isTuple());
    const auto tuple = value.toTuple();
    TORCH_CHECK(tuple->elements().size() == 2);
    std::vector<c10::IValue> values;
    values.reserve(2);
    for (int i = 0; i < 2; ++i) {
      const auto& tensor = tuple->elements()[i].toTensor();
      values.emplace_back(
          tensor.to(at::Device(at::kCPU), /* non_blocking */ true));
    }
    moved.insert(key, c10::ivalue::Tuple::create(std::move(values)));
  }
  return moved;
}

} // namespace torchrec
