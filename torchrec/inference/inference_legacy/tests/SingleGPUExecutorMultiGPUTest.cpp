/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <folly/futures/Future.h>
#include <gtest/gtest.h>
#include <multipy/runtime/deploy.h>
#include <torch/cuda.h> // @manual
#include <torch/script.h>
#include <torch/torch.h> // @manual
#include "torchrec/inference/Observer.h"
#include "torchrec/inference/SingleGPUExecutor.h"
#include "torchrec/inference/Types.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

const char* path(const char* envname, const char* path) {
  const char* e = std::getenv(envname);
  return e ? e : path;
}

std::vector<c10::IValue> get_input_example(
    torch::deploy::InterpreterSession& model_interpreter_session) {
  auto eg = model_interpreter_session.self
                .attr("load_pickle")({"model", "example.pkl"})
                .toIValue();
  return eg.toTupleRef().elements();
}

void assert_tensors_eq(const at::Tensor& expected, const at::Tensor& got) {
  ASSERT_TRUE(expected.allclose(got, 1e-03, 1e-05));
}

c10::IValue execute(
    torchrec::SingleGPUExecutor& executor,
    const std::string& methodName,
    std::vector<c10::IValue> args) {
  folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
  auto future = promise.getSemiFuture();

  executor.schedule(
      std::make_shared<torchrec::PredictionBatch>(
          methodName, std::move(args), std::move(promise)));
  return std::move(future).get()->predictions;
}

TEST(TorchDeployGPUTest, SimpleModel_multiGPU) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "Test is skipped as it requires CUDA.";
  }

  const size_t numGpu = torch::cuda::device_count();
  if (numGpu <= 1) {
    GTEST_SKIP() << "Test is skipped as it requires > 1 CUDA devices, found:"
                 << numGpu;
  }

  const char* model_filename = path("TORCH_PACKAGE_SIMPLE", "/tmp/simple");

  auto device = c10::Device(c10::kCUDA, 0);

  auto manager =
      std::make_shared<torch::deploy::InterpreterManager>(2 * numGpu);
  torch::deploy::Package package = manager->loadPackage(model_filename);

  std::vector<torch::deploy::ReplicatedObj> models;
  torch::deploy::ReplicatedObj model_control;
  const size_t gpu_rank_control = 0;

  {
    auto I = package.acquireSession();

    auto pyModel = I.fromMovable(package.loadPickle("model", "model.pkl"));

    for (size_t i = 0; i < numGpu; i++) {
      auto model = I.createMovable(
          pyModel.attr("to")(c10::IValue(c10::Device(c10::kCUDA, i))));
      if (i == gpu_rank_control) {
        model_control = model;
      }
      models.push_back(std::move(model));
    }
  }

  std::vector<std::unique_ptr<torchrec::SingleGPUExecutor>> workExecutors;
  for (size_t i = 0; i < numGpu; i++) {
    const std::vector<size_t> interp_idxs = {static_cast<size_t>(i)};
    workExecutors.push_back(
        std::make_unique<torchrec::SingleGPUExecutor>(
            manager,
            torchrec::SingleGPUExecutor::ExecInfos{{i, numGpu + i, models[i]}},
            numGpu));
  }

  std::vector<torchrec::SingleGPUExecutor::ExecInfo> execInfos;
  for (size_t i = 0; i < numGpu; i++) {
    execInfos.push_back({i, numGpu + i, models[i]});
  }

  auto controlExecutor =
      std::make_unique<torchrec::SingleGPUExecutor>(manager, execInfos, numGpu);

  std::vector<at::IValue> example_inputs;
  {
    auto I = package.acquireSession();
    example_inputs = get_input_example(I);
  }
  auto example_input0 = example_inputs[0].toTensor();
  auto expected_forward0 = example_input0 + at::ones(example_input0.sizes());

  for (size_t i = 0; i < numGpu; i++) {
    auto result =
        execute(*workExecutors[i], "forward", example_inputs).toTensor();
    assert_tensors_eq(expected_forward0, result);
  }

  execute(*controlExecutor, "set_weight", {at::zeros(example_input0.sizes())});

  auto checkFn = [&](size_t set_weight_count) {
    for (size_t i = 0; i < numGpu; i++) {
      auto result =
          execute(*workExecutors[i], "forward", example_inputs).toTensor();
      if (i < set_weight_count) {
        assert_tensors_eq(example_input0, result);
      } else {
        assert_tensors_eq(expected_forward0, result);
      }
    }
  };
  checkFn(1u);

  execute(*controlExecutor, "set_weight", {at::zeros(example_input0.sizes())});
  checkFn(2u);
}
