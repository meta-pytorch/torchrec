/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/BatchingQueue.h" // @manual

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <ATen/Functions.h> // @manual
#include <ATen/core/Dict.h>
#include <ATen/record_function.h> // @manual
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <fmt/format.h>
#include <folly/ExceptionString.h>
#include <folly/Lazy.h>
#include <folly/MPMCQueue.h>
#include <folly/Range.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/io/Cursor.h>
#include <glog/logging.h>

#include "torchrec/inference/ExceptionHandler.h"
#include "torchrec/inference/Observer.h"
#include "torchrec/inference/ResourceManager.h"
#include "torchrec/inference/Types.h"

using namespace std::chrono_literals;

DEFINE_bool(
    batching_queue_use_high_pri_stream,
    false,
    "Use high priority CUDA stream in batching");

namespace torchrec {

BatchingQueue::BatchingQueue(
    std::vector<BatchQueueCb> cbs,
    const Config& config,
    int worldSize,
    std::unique_ptr<IBatchingQueueObserver> observer,
    std::shared_ptr<ResourceManager> resourceManager)
    : config_(config),
      cbs_(std::move(cbs)),
      stopping_(false),
      worldSize_(worldSize),
      observer_(std::move(observer)),
      resourceManager_(std::move(resourceManager)) {
  CHECK(observer_ != nullptr);
  for (const auto& [_, metadata] : config_.batchingMetadata) {
    if (batchingFuncs_.contains(metadata.type)) {
      continue;
    }
    batchingFuncs_[metadata.type] =
        TorchRecBatchingFuncRegistry()->Create(metadata.type);
  }
  for (int i = 0; i < worldSize_; i++) {
    auto queue = std::make_shared<folly::MPMCQueue<BatchingQueueEntry>>(
        folly::MPMCQueue<BatchingQueueEntry>(1000));
    batchingQueues_.push_back(std::move(queue));
  }
  rejectionExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
      config_.numExceptionThreads);
  batchingThread_ = std::thread([&] {
    folly::setThreadName("CreateBatch");
    createBatch();
  });
  for (int i = 0; i < worldSize_; ++i) {
    for (int j = 0; j < config_.numMemPinnerThreads; ++j) {
      memPinnerThreads_.emplace_back([&, i, j] {
        c10::InferenceMode guard;
        folly::setThreadName(fmt::format("MemoryPinner-{}-GPU-{}", j, i));
        pinMemory(i);
      });
    }
  }
}

void BatchingQueue::add(
    std::shared_ptr<PredictionRequest> request,
    folly::Promise<std::unique_ptr<PredictionResponse>> promise) {
  CHECK_GT(request->batch_size, 0);
  const auto addedTime = std::chrono::steady_clock::now();
  requestQueue_.withWLock([request = std::move(request),
                           promise = std::move(promise),
                           addedTime = addedTime](auto& queue) mutable {
    const auto batchSize = request->batch_size;
    queue.push(
        QueryQueueEntry{
            std::move(request),
            RequestContext{
                batchSize,
                std::move(promise),
                folly::RequestContext::saveContext()},
            addedTime});
  });
}

void BatchingQueue::stop() {
  stopping_ = true;
  // TODO: properly drain the queue before stopping the threads.
  batchingThread_.join();
  for (auto& thread : memPinnerThreads_) {
    thread.join();
  }
}

void BatchingQueue::createBatch() {
  std::optional<std::chrono::time_point<std::chrono::steady_clock>> startTime;
  size_t batchSize = 0;
  std::vector<std::shared_ptr<PredictionRequest>> requests;
  std::vector<RequestContext> contexts;
  int roundRobinIdx = 0;

  while (!stopping_) {
    bool full = false;

    requestQueue_.withWLock([&](auto& queue) {
      while (!queue.empty()) {
        auto& front = queue.front();
        if (std::chrono::steady_clock::now() - front.addedTime >=
            config_.queueTimeout) {
          observer_->addBatchingQueueTimeoutCount(1);
          rejectionExecutor_->add(
              [promise = std::move(front.context.promise)]() mutable {
                handleRequestException<GPUOverloadException>(
                    promise, "Batching queue timeout");
              });
          queue.pop();
          continue;
        }

        if (!startTime) {
          startTime = front.addedTime;
        }

        if (batchSize > 0 &&
            (batchSize + front.request->batch_size > config_.maxBatchSize)) {
          full = true;
          break;
        }

        auto& context = contexts.emplace_back(std::move(front.context));
        folly::RequestContext::setContext(context.follyRequestContext);
        requests.push_back(std::move(front.request));
        batchSize += requests.back()->batch_size;
        queue.pop();
      }
    });

    if (full ||
        (startTime &&
         (std::chrono::steady_clock::now() - *startTime >=
          config_.batchingInterval))) {
      const auto requestsCount = requests.size();

      batchingQueues_[roundRobinIdx++]->blockingWrite(
          BatchingQueueEntry{
              .requests = std::move(requests),
              .contexts = std::move(contexts),
              .addedTime = *startTime});

      observer_->addRequestsCount(requestsCount);
      observer_->recordBatchCreationLatency(
          getTimeElapsedMS(*startTime).count());

      startTime.reset();
      batchSize = 0;
      roundRobinIdx %= worldSize_;
      requests.clear();
      contexts.clear();
    }

    folly::RequestContext::setContext(nullptr);

    if (!full) {
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void BatchingQueue::pinMemory(int gpuIdx) {
  at::cuda::CUDAGuard deviceGuard(gpuIdx);
  at::cuda::CUDAStreamGuard streamGuard(
      at::cuda::getStreamFromPool(
          /* isHighPriority */ FLAGS_batching_queue_use_high_pri_stream));
  if (config_.warmupFn) {
    config_.warmupFn();
  }

  while (!stopping_) {
    BatchingQueueEntry entry;
    if (!batchingQueues_[gpuIdx]->tryReadUntil(
            std::chrono::steady_clock::now() + 10ms, entry)) {
      continue;
    }

    auto& requests = entry.requests;
    auto& contexts = entry.contexts;

    try {
      if (!requests.empty() || !contexts.empty()) {
        RECORD_USER_SCOPE("PinMemory");

        if (!contexts.empty()) {
          folly::RequestContext::setContext(contexts[0].follyRequestContext);
        }
        // Combine data.
        size_t combinedBatchSize = 0;
        for (auto i : c10::irange(requests.size())) {
          combinedBatchSize += requests[i]->batch_size;
        }

        auto batchOffsetsLazy =
            folly::lazy<std::function<at::Tensor()>>([&]() -> at::Tensor {
              size_t batchSize = 0;
              auto batchOffsets = at::empty(
                  {static_cast<long>(requests.size() + 1)},
                  at::TensorOptions().dtype(at::kInt).pinned_memory(true));
              auto batchOffsetsAcc = batchOffsets.accessor<int32_t, 1>();
              batchOffsetsAcc[0] = 0;
              for (auto i : c10::irange(requests.size())) {
                batchSize += requests[i]->batch_size;
                batchOffsetsAcc[i + 1] = batchSize;
              }
              return batchOffsets;
            });

        auto batchItemsLazy =
            folly::lazy<std::function<at::Tensor()>>([&]() -> at::Tensor {
              auto batchItems = at::empty(
                  {static_cast<int64_t>(combinedBatchSize)},
                  at::TensorOptions().dtype(at::kInt).pinned_memory(true));
              auto batchItemsAcc = batchItems.accessor<int32_t, 1>();
              auto batchOffsetsAcc = batchOffsetsLazy().accessor<int32_t, 1>();
              for (auto i = 0; i < requests.size(); ++i) {
                auto start = batchOffsetsAcc[i];
                auto end = batchOffsetsAcc[i + 1];
                for (auto j = start; j < end; ++j) {
                  batchItemsAcc[j] = i;
                }
              }
              return batchItems;
            });

        c10::impl::GenericDict forwardArgs(
            at::StringType::get(), at::AnyType::get());
        auto combineForwardArgs =
            [&](std::unordered_map<std::string, c10::IValue> map) {
              for (auto& [key, value] : map) {
                CHECK(!forwardArgs.contains(key));
                forwardArgs.insert(key, std::move(value));
              }
            };

        auto elapsed = getTimeElapsedMS(entry.addedTime);

        observer_->recordBatchingQueueLatency(elapsed.count());

        if (resourceManager_ != nullptr) {
          auto slack = std::chrono::duration_cast<std::chrono::milliseconds>(
              config_.queueTimeout - elapsed);
          auto gpuFree = slack.count() > 0
              ? resourceManager_->occupyDevice(gpuIdx, slack)
              : false;

          if (!gpuFree) {
            // A device could not be chosen in time. Time out.
            observer_->addGPUBusyCount(1);
            rejectionExecutor_->add([ctxs = std::move(contexts)]() mutable {
              handleBatchException<GPUOverloadException>(
                  ctxs, "All GPUs are busy. Batching queue timeout.");
            });
            continue;
          }
        }

        // When resourceManagerGuard goes out of scope, the number of
        // oustanding batches associated with a gpu is decremented. Should be
        // defined before batching functions are called in case they throw
        // an exception.
        auto resourceManagerGuard = resourceManager_ != nullptr
            ? std::make_unique<ResourceManagerGuard>(resourceManager_, gpuIdx)
            : nullptr;

        for (auto& [featureName, metadata] : config_.batchingMetadata) {
          const auto batchingFuncStart = std::chrono::steady_clock::now();
          combineForwardArgs(
              batchingFuncs_[metadata.type]->batch(
                  featureName,
                  requests,
                  combinedBatchSize,
                  batchOffsetsLazy,
                  metadata.device == "cpu" ? c10::Device(c10::kCPU)
                                           : c10::Device(c10::kCUDA, gpuIdx),
                  batchItemsLazy));
          observer_->recordBatchingFuncLatency(
              getTimeElapsedMS(batchingFuncStart).count(), metadata.type);
        }

        // The batch is moved to the GPUExecutor, which can either
        // throw an exception or complete the batch. Move resourceManagerGuard
        // with the batch so when all references to the batch go to 0,
        // num outstanding requests for the gpu decrements.
        auto batch = std::make_shared<PredictionBatch>(
            combinedBatchSize,
            std::move(forwardArgs),
            std::move(contexts),
            std::move(resourceManagerGuard));

        auto createEvent = [&]() {
          return Event(
              std::make_unique<at::cuda::CUDAEvent>(
                  cudaEventBlockingSync | cudaEventDisableTiming)
                  .release(),
              [](at::cuda::CUDAEvent* event) { delete event; });
        };

        batch->event = config_.eventCreationFn ? config_.eventCreationFn(gpuIdx)
                                               : createEvent();
        batch->event->record();

        observer_->observeBatchCompletion(batch->size(), batch->batchSize);

        cbs_[gpuIdx](batch);

        // unset request tracking
        folly::RequestContext::setContext(nullptr);
      }
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Error batching requests, ex: " << folly::exceptionStr(ex);
      for (auto& ctx : contexts) {
        rejectionExecutor_->add([promise = std::move(ctx.promise)]() mutable {
          handleRequestException<TorchrecException>(
              promise, "Error during batching requests");
        });
      }
    }
  }
}

BatchingQueue::~BatchingQueue() {
  stop();
}

} // namespace torchrec
