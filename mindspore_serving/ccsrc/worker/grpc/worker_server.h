/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_SERVING_WORKER_WORKER_SERVER_H
#define MINDSPORE_SERVING_WORKER_WORKER_SERVER_H

#include <memory>
#include <string>
#include "common/serving_common.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "common/grpc_async_server.h"
#include "worker/grpc/worker_process.h"
#include "worker/distributed_worker/distributed_servable.h"

namespace mindspore {
namespace serving {

template <class Derived>
class WorkerServiceContext : public GrpcAsyncServiceContext<MSWorkerImpl, proto::MSWorker::AsyncService, Derived> {
 public:
  WorkerServiceContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : GrpcAsyncServiceContext<MSWorkerImpl, proto::MSWorker::AsyncService, Derived>(service_impl, async_service, cq) {
  }
  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;
};

class WorkerPredictContext : public WorkerServiceContext<WorkerPredictContext> {
 public:
  WorkerPredictContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerPredictContext() = default;

  void StartEnqueueRequest() override { async_service_->RequestPredict(&ctx_, &request_, &responder_, cq_, cq_, this); }

  void HandleRequest() override {
    MSI_TIME_STAMP_START(WorkerRequestHandle)
    PredictOnFinish on_finish = [this, time_start_WorkerRequestHandle]() {
      responder_.Finish(response_, grpc::Status::OK, this);
      MSI_TIME_STAMP_END(WorkerRequestHandle)
    };
    service_impl_->PredictAsync(&ctx_, &request_, &response_, on_finish);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder_;
  proto::PredictRequest request_;
  proto::PredictReply response_;
};

class WorkerExitContext : public WorkerServiceContext<WorkerPredictContext> {
 public:
  WorkerExitContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerExitContext() = default;

  void StartEnqueueRequest() override { async_service_->RequestExit(&ctx_, &request_, &responder_, cq_, cq_, this); }

  void HandleRequest() override { service_impl_->Exit(&ctx_, &request_, &response_); }

 private:
  grpc::ServerAsyncResponseWriter<proto::ExitReply> responder_;
  proto::ExitRequest request_;
  proto::ExitReply response_;
};

class WorkerGrpcServer : public GrpcAsyncServer<proto::MSWorker::AsyncService> {
 public:
  WorkerGrpcServer() : GrpcAsyncServer<proto::MSWorker::AsyncService>() {}
  void EnqueueRequests() override { WorkerPredictContext::EnqueueRequest(&service_impl_, &svc_, cq_.get()); }

 protected:
  MSWorkerImpl service_impl_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H
