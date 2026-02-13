/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/GPUExecutor.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string_view>

#include <c10/cuda/CUDAGuard.h>
#include <fmt/format.h>
#include <folly/MPMCQueue.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/io/async/Request.h>
#include <folly/stop_watch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/csrc/autograd/profiler.h> // @manual

// remove this after we switch over to multipy externally for torchrec
#ifdef FBCODE_CAFFE2
#include <multipy/runtime/deploy.h> // @manual
#else
#include <torch/csrc/deploy/deploy.h> // @manual
#endif

#include "torchrec/inference/ExceptionHandler.h"
#include "torchrec/inference/Observer.h"
#include "torchrec/inference/Types.h"

DEFINE_int32(copy_timeout, 500, "");

DEFINE_bool(
    emit_nsys_nvtx,
    false,
    "emit NVTX markers/ranges to be visualized in NSight Systems");

DEFINE_bool(gpu_executor_use_high_pri_stream_main_device, false, "");

DEFINE_bool(gpu_executor_use_high_pri_stream_peer_device, false, "");

DEFINE_bool(gpu_executor_use_high_pri_stream_d2h, false, "");

namespace torchrec {

namespace {
// Enable NVTX tracing for the caller thread if the flag is set.
void enable_nvtx_tracing() {
  thread_local static bool emit = false;
  if (FLAGS_emit_nsys_nvtx && emit == false) {
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::NVTX, false, false));
    emit = true;
  }
}
} // namespace

GPUExecutor::GPUExecutor(
    std::shared_ptr<torch::deploy::InterpreterManager> manager,
    torch::deploy::ReplicatedObj model,
    size_t rank,
    size_t worldSize,
    std::shared_ptr<torchrec::ResultSplitFunc> func,
    std::chrono::milliseconds queueTimeout,
    std::shared_ptr<IGPUExecutorObserver> observer,
    std::function<void()> warmupFn,
    std::optional<size_t> numThreadsPerGPU,
    std::unique_ptr<GCConfig> gcConfig)
    : manager_(manager),
      model_(std::move(model)),
      rank_(rank),
      worldSize_(worldSize),
      batches_(10'000),
      resultSplitFunc_(func),
      queueTimeout_(queueTimeout),
      observer_(observer),
      warmupFn_(std::move(warmupFn)),
      numThreadsPerGPU_(
          numThreadsPerGPU.has_value()
              ? *numThreadsPerGPU
              : manager_->allInstances().size() / worldSize_),
      gcConfig_(std::move(gcConfig)) {
  CHECK(observer_ != nullptr);
  CHECK(gcConfig_ != nullptr);

  at::cuda::CUDAGuard guard(rank_);

  rejectionExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(2 * numThreadsPerGPU_);
  for (int i = 0; i < numThreadsPerGPU_; ++i) {
    auto threadId = rank * numThreadsPerGPU_ + i;

    LOG(INFO) << "Starting Thread " << i << " for Model Shard Rank " << rank_
              << ", as Global thread: " << threadId;

    if (gcConfig_->optimizationEnabled) {
      gcConfig_->threadIdToNumForwards[threadId] = 0;
      // Freeze all python objects in each interpreter
      auto session =
          model_.acquireSession(&manager_->allInstances().at(threadId));
      session.global("gc", "freeze")(at::ArrayRef<torch::deploy::Obj>());
    }

    processThreads_.emplace_back([this, threadId] {
      if (FLAGS_emit_nsys_nvtx) {
        enable_nvtx_tracing();
      }
      process(threadId);
    });
  }

  // Acquire sessionn in main thread for interpreter 0 to avoid deadlock in
  // torch deploy.
  {
    std::lock_guard<std::mutex> lock(warmUpAcquireSessionMutex_);
    LOG(INFO)
        << " - pre-acquire deploy session of loading model: interpreter 0";
    auto start = std::chrono::steady_clock::now();
    model_.acquireSession(&manager_->allInstances().at(0));
    LOG(INFO) << "   - finished pre-acquire deploy session, interpreter 0, by "
              << getTimeElapsedMS(start).count() / 1000 << "s";
  }

  std::unique_lock<std::mutex> lock(warmUpMutex_);
  warmUpCV_.wait(lock, [&] { return warmUpCounter_ == numThreadsPerGPU_ - 1; });

  completionExecutor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(2 * numThreadsPerGPU_);
}

GPUExecutor::~GPUExecutor() {
  for (int i = 0; i < processThreads_.size(); ++i) {
    batches_.blockingWrite(nullptr);
  }
  for (auto& thread : processThreads_) {
    thread.join();
  }
  completionExecutor_->join();

  std::shared_ptr<PredictionBatch> batch;
  while (batches_.readIfNotEmpty(batch)) {
    rejectionExecutor_->add([batch = std::move(batch)]() {
      handleBatchException<PredictionException>(
          batch->contexts, "Server shutdown");
    });
  }
}

void GPUExecutor::callback(std::shared_ptr<PredictionBatch> batch) {
  batches_.blockingWrite(std::move(batch));
}

void GPUExecutor::process(int idx) {
  folly::setThreadName(
      fmt::format("GPU-{}: Thread-{}", rank_, idx % numThreadsPerGPU_));

  c10::InferenceMode inferenceModeGuard;
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < worldSize_; ++i) {
    streams.push_back(
        at::cuda::getStreamFromPool(
            /* isHighPriority */ i == rank_
                ? FLAGS_gpu_executor_use_high_pri_stream_main_device
                : FLAGS_gpu_executor_use_high_pri_stream_peer_device,
            i));
  }
  at::cuda::CUDAMultiStreamGuard streamGuard(streams);
  at::cuda::CUDAGuard deviceGuard(rank_);
  auto d2hStream = at::cuda::getStreamFromPool(
      /* isHighPriority */ FLAGS_gpu_executor_use_high_pri_stream_d2h, rank_);

  if (warmupFn_) {
    warmupFn_();
  }

  if (idx != 0) {
    std::lock_guard<std::mutex> lock(warmUpAcquireSessionMutex_);
    LOG(INFO) << " - Pre-acquire deploy session of loading model, interpreter "
              << idx;
    auto start = std::chrono::steady_clock::now();
    model_.acquireSession(&manager_->allInstances().at(idx));
    {
      std::lock_guard<std::mutex> warmUpLock(warmUpMutex_);
      warmUpCounter_++;
      warmUpCV_.notify_one();
    }
    LOG(INFO) << "   - finished pre-acquire deploy session, interpreter " << idx
              << ", by " << getTimeElapsedMS(start).count() / 1000 << "s";
  }

  while (true) {
    std::shared_ptr<PredictionBatch> batch;
    batches_.blockingRead(batch);
    if (batch == nullptr) {
      // shutdown
      break;
    }

    if (batch->batchSize == 0) {
      continue;
    }

    if (!batch->contexts.empty()) {
      folly::RequestContext::setContext(batch->contexts[0].follyRequestContext);
    }

    auto timeInQueue = getTimeElapsedMS(batch->enqueueTime);
    observer_->recordQueueLatency(timeInQueue.count());

    if (timeInQueue >= queueTimeout_) {
      observer_->addQueueTimeoutCount(1);
      rejectionExecutor_->add([batch = std::move(batch)]() {
        handleBatchException<GPUExecutorOverloadException>(
            batch->contexts, "GPUExecutor queue timeout");
      });

      continue;
    }

    // Free session to avoid accumulating too many PyObjects.
    auto model = model_.acquireSession(&manager_->allInstances().at(idx));
    at::IValue predictions;

    LOG_EVERY_N(INFO, 10000)
        << "GPU " << rank_ << " is running batch size " << batch->batchSize
        << ", avg request size " << batch->batchSize / batch->contexts.size();

    std::string exWhat;
    try {
      RECORD_USER_SCOPE("Forward");
      // Block current stream until H2D finishes.
      batch->event->block(streams[rank_]);

      auto forwardStart = std::chrono::steady_clock::now();

      // Disable automatic garbage collection
      if (gcConfig_->optimizationEnabled) {
        model.global("gc", "disable")(at::ArrayRef<torch::deploy::Obj>());
      }

      predictions = model.self.attr("__call__")({std::move(batch->forwardArgs)})
                        .toIValue();

      // Manually call Python's garbage collector
      if (gcConfig_->optimizationEnabled) {
        gcConfig_->threadIdToNumForwards[idx] += 1;
        if (gcConfig_->threadIdToNumForwards[idx] % gcConfig_->collectionFreq ==
            0) {
          model.global("gc", "collect")(at::ArrayRef<torch::deploy::Obj>());
        }
        // Report gc stats
        if (gcConfig_->threadIdToNumForwards[idx] %
                gcConfig_->statReportingFreq ==
            0) {
          reportGCStats(
              model
                  .global("gc", "get_stats")(at::ArrayRef<torch::deploy::Obj>())
                  .toIValue());
        }
      }

      observer_->recordPredictionLatency(
          getTimeElapsedMS(forwardStart).count());
    } catch (const std::exception& ex) {
      // The observer will record this in the completion executor. Don't
      // observe twice.
      LOG_EVERY_N(ERROR, 100) << "Exception during predict, msg: " << ex.what();
      exWhat = ex.what();
    }

    batch->event->record();

    completionExecutor_->add(
        // Can not bind the method directly because of the unique_ptr of item.
        [this,
         batch = std::move(batch),
         predictions = std::move(predictions),
         resultSplitFunc = resultSplitFunc_,
         rank = rank_,
         d2hStream = d2hStream,
         observer = observer_.get(),
         exWhat = exWhat]() mutable {
          RECORD_USER_SCOPE("CompletionStage");
          c10::InferenceMode imGuard;

          batch->event->block(d2hStream);

          at::cuda::CUDAStreamGuard streamGuard(d2hStream);
          at::cuda::CUDAGuard deviceGuard(rank);

          if (!predictions.isNone()) {
            auto d2hStart = std::chrono::steady_clock::now();

            predictions = resultSplitFunc->moveToHost(predictions);
            batch->event->record();
            // Wait for D2H to finish.
            batch->event->synchronize();

            observer->recordDeviceToHostLatency(
                getTimeElapsedMS(d2hStart).count(), resultSplitFunc->name());
          }

          constexpr std::string_view gpuExceptionContext =
              "GPUExecutor prediction exception, ";
          if (predictions.isNone()) {
            observer->addPredictionExceptionCount(1);
            rejectionExecutor_->add([contexts = std::move(batch->contexts),
                                     gpuExceptionContext = gpuExceptionContext,
                                     exWhat = exWhat]() mutable {
              handleBatchException<TorchDeployException>(
                  contexts,
                  std::string(
                      gpuExceptionContext.begin(), gpuExceptionContext.end()) +
                      exWhat);
            });
          } else {
            size_t offset = 0;
            auto rsfStart = std::chrono::steady_clock::now();
            for (auto& context : batch->contexts) {
              CHECK(!predictions.isNone());
              CHECK_LT(offset, batch->batchSize);
              auto response = std::make_unique<PredictionResponse>();
              response->batchSize = context.batchSize;
              response->predictions = resultSplitFunc->splitResult(
                  predictions, offset, context.batchSize, batch->batchSize);
              context.promise.setValue(std::move(response));
              offset += context.batchSize;
            }
            observer->recordResultSplitLatency(
                getTimeElapsedMS(rsfStart).count(), resultSplitFunc->name());
            CHECK_EQ(offset, batch->batchSize);
            observer->addBatchesProcessedCount(1);
          }
          observer->recordTotalLatency(
              getTimeElapsedMS(batch->enqueueTime).count());
        });

    // reset request tracking
    folly::RequestContext::setContext(nullptr);
  }
}

void GPUExecutor::reportGCStats(c10::IValue stats) {
  const auto generationsList = stats.toList();
  for (const auto generationId : c10::irange(generationsList.size())) {
    const auto& collectionsDict =
        generationsList.get(generationId).toGenericDict();
    for (auto& entry : collectionsDict) {
      const auto& key = entry.key();
      const auto& value = entry.value();

      auto stat_indicator =
          "gc_gen_" + std::to_string(generationId) + "_" + key.toStringRef();
      LOG(INFO) << "GC stat indicator: " << stat_indicator;
      gcConfig_->observer->addCount(value.toInt(), stat_indicator);
    }
  }
}

} // namespace torchrec
