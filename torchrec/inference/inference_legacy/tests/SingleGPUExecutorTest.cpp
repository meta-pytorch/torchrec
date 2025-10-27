/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/SingleGPUExecutor.h"
#include <folly/futures/Future.h>
#include <gtest/gtest.h>
#include <multipy/runtime/deploy.h>
#include <torch/cuda.h> // @manual
#include <torch/script.h>
#include <torch/torch.h> // @manual
#include "torchrec/inference/Observer.h"
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

TEST(TorchDeployGPUTest, SimpleModelSingleGPU) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "Test is skipped as it requires CUDA.";
  }

  const char* model_filename = path("TORCH_PACKAGE_SIMPLE", "/tmp/simple");

  auto device = c10::Device(c10::kCUDA, 0);

  auto manager = std::make_shared<torch::deploy::InterpreterManager>(1);
  torch::deploy::Package p = manager->loadPackage(model_filename);
  auto model = p.loadPickle("model", "model.pkl");
  {
    auto M = model.acquireSession();
    M.self.attr("to")({device});
  }

  std::vector<at::IValue> example_inputs;
  {
    auto I = p.acquireSession();
    example_inputs = get_input_example(I);
  }

  auto executor = std::make_unique<torchrec::SingleGPUExecutor>(
      manager,
      torchrec::SingleGPUExecutor::ExecInfos{{0u, 0u, std::move(model)}},
      1u);

  {
    auto inputs = example_inputs;

    // Calling forward on freshly loaded model
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();

    executor->schedule(
        std::make_shared<torchrec::PredictionBatch>(
            "forward", inputs, std::move(promise)));

    auto forward_result_0 = std::move(future).get()->predictions.toTensor();
    auto expected = example_inputs[0].toTensor() + at::ones({10, 20});
    assert_tensors_eq(expected, forward_result_0);
  }

  {
    // Calling set_weight
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();

    std::vector<c10::IValue> set_weight_inputs = {at::zeros({10, 20})};

    executor->schedule(
        std::make_shared<torchrec::PredictionBatch>(
            "set_weight", set_weight_inputs, std::move(promise)));

    auto predictionResponse = std::move(future).get();
  }

  {
    auto inputs = example_inputs;
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();

    executor->schedule(
        std::make_shared<torchrec::PredictionBatch>(
            "forward", inputs, std::move(promise)));

    auto forward_result_1 = std::move(future).get()->predictions.toTensor();
    auto expected = example_inputs[0].toTensor();
    assert_tensors_eq(expected, forward_result_1);
  }
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

TEST(TorchDeployGPUTest, NestedModelSingleGPU) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "Test is skipped as it requires CUDA.";
  }

  const char* model_filename = path("TORCH_PACKAGE_NESTED", "/tmp/nested");

  auto device = c10::Device(c10::kCUDA, 0);

  auto manager = std::make_shared<torch::deploy::InterpreterManager>(1);
  torch::deploy::Package p = manager->loadPackage(model_filename);
  auto model = p.loadPickle("model", "model.pkl");
  {
    auto M = model.acquireSession();
    M.self.attr("to")({device});
  }

  std::vector<at::IValue> example_inputs;
  {
    auto I = p.acquireSession();
    example_inputs = get_input_example(I);
  }
  auto example_input0 = example_inputs[0].toTensor();

  auto executor = std::make_unique<torchrec::SingleGPUExecutor>(
      manager,
      torchrec::SingleGPUExecutor::ExecInfos{{0u, 0u, std::move(model)}},
      1u);

  auto expected_forward0 = example_input0 + at::ones(example_input0.sizes());

  auto inputs = example_inputs;
  auto result = execute(*executor, "simple.forward", example_inputs).toTensor();
  assert_tensors_eq(expected_forward0, result);
}

class TestSingleGPUExecutorObserver
    : public torchrec::EmptySingleGPUExecutorObserver {
 public:
  double requestCount = 0.f;
  void addRequestsCount(uint32_t value) override {
    requestCount += value;
  }
};

TEST(TorchDeployGPUTest, SimpleModelSingleGPUObserver) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "Test is skipped as it requires CUDA.";
  }

  const char* model_filename = path("TORCH_PACKAGE_NESTED", "/tmp/simple");

  auto device = c10::Device(c10::kCUDA, 0);

  auto manager = std::make_shared<torch::deploy::InterpreterManager>(1);
  torch::deploy::Package p = manager->loadPackage(model_filename);
  auto model = p.loadPickle("model", "model.pkl");
  {
    auto M = model.acquireSession();
    M.self.attr("to")({device});
  }

  std::vector<at::IValue> example_inputs;
  {
    auto I = p.acquireSession();
    example_inputs = get_input_example(I);
  }
  auto example_input0 = example_inputs[0].toTensor();

  auto observer = std::make_shared<TestSingleGPUExecutorObserver>();

  auto executor = std::make_unique<torchrec::SingleGPUExecutor>(
      manager,
      torchrec::SingleGPUExecutor::ExecInfos{{0u, 0u, std::move(model)}},
      1u /* numGPUs */,
      observer);

  auto expected_forward0 = example_input0 + at::ones(example_input0.sizes());

  auto inputs = example_inputs;
  auto result = execute(*executor, "forward", example_inputs).toTensor();
  assert_tensors_eq(expected_forward0, result);
  ASSERT_EQ(observer->requestCount, 1.0f);
}
