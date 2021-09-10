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

#ifndef MINDSPORE_SERVING_WORKER_DISTRIBUTED_WORKER_SERVER_H
#define MINDSPORE_SERVING_WORKER_DISTRIBUTED_WORKER_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <string>
#include "common/serving_common.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "common/grpc_async_server.h"
#include "worker/grpc/worker_process.h"
#include "worker/grpc/worker_server.h"
#include "worker/distributed_worker/distributed_process/distributed_process.h"

namespace mindspore {
namespace serving {

template <class Derived>
class DistributedServiceContext
    : public GrpcAsyncServiceContext<MSDistributedImpl, proto::MSDistributedWorker::AsyncService, Derived> {
 public:
  DistributedServiceContext(MSDistributedImpl *service_impl, proto::MSDistributedWorker::AsyncService *async_service,
                            grpc::ServerCompletionQueue *cq)
      : GrpcAsyncServiceContext<MSDistributedImpl, proto::MSDistributedWorker::AsyncService, Derived>(
          service_impl, async_service, cq) {}
  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;
};

// Service Implement
class WorkerAgentRegisterContext : public DistributedServiceContext<WorkerAgentRegisterContext> {
 public:
  WorkerAgentRegisterContext(MSDistributedImpl *service_impl, proto::MSDistributedWorker::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext<WorkerAgentRegisterContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentRegisterContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestAgentRegister(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->AgentRegister(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentRegisterReply> responder_;
  proto::AgentRegisterRequest request_;
  proto::AgentRegisterReply response_;
};

class WorkerAgentExitContext : public DistributedServiceContext<WorkerAgentExitContext> {
 public:
  WorkerAgentExitContext(MSDistributedImpl *service_impl, proto::MSDistributedWorker::AsyncService *async_service,
                         grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext<WorkerAgentExitContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentExitContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestAgentExit(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->AgentExit(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentExitReply> responder_;
  proto::AgentExitRequest request_;
  proto::AgentExitReply response_;
};

class WorkerAgentFailedContext : public DistributedServiceContext<WorkerAgentFailedContext> {
 public:
  WorkerAgentFailedContext(MSDistributedImpl *service_impl, proto::MSDistributedWorker::AsyncService *async_service,
                           grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext<WorkerAgentFailedContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentFailedContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestAgentFailed(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->AgentFailed(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentFailedReply> responder_;
  proto::AgentFailedRequest request_;
  proto::AgentFailedReply response_;
};

class WorkerAgentConfigAcquireContext : public DistributedServiceContext<WorkerAgentConfigAcquireContext> {
 public:
  WorkerAgentConfigAcquireContext(MSDistributedImpl *service_impl,
                                  proto::MSDistributedWorker::AsyncService *async_service,
                                  grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext<WorkerAgentConfigAcquireContext>(service_impl, async_service, cq),
        responder_(&ctx_) {}

  ~WorkerAgentConfigAcquireContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestAgentConfigAcquire(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->AgentConfigAcquire(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentConfigAcquireReply> responder_;
  proto::AgentConfigAcquireRequest request_;
  proto::AgentConfigAcquireReply response_;
};

class WorkerPingContext : public DistributedServiceContext<WorkerPingContext> {
 public:
  WorkerPingContext(MSDistributedImpl *service_impl, proto::MSDistributedWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext<WorkerPingContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerPingContext() = default;

  void StartEnqueueRequest() override { async_service_->RequestPing(&ctx_, &request_, &responder_, cq_, cq_, this); }

  void HandleRequest() override {
    grpc::Status status = service_impl_->Ping(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PingReply> responder_;
  proto::PingRequest request_;
  proto::PingReply response_;
};

class WorkerPongContext : public DistributedServiceContext<WorkerPongContext> {
 public:
  WorkerPongContext(MSDistributedImpl *service_impl, proto::MSDistributedWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext<WorkerPongContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerPongContext() = default;

  void StartEnqueueRequest() override { async_service_->RequestPong(&ctx_, &request_, &responder_, cq_, cq_, this); }

  void HandleRequest() override {
    grpc::Status status = service_impl_->Pong(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PongReply> responder_;
  proto::PongRequest request_;
  proto::PongReply response_;
};

class DistributedWorkerGrpcServer : public GrpcAsyncServer<proto::MSDistributedWorker::AsyncService> {
 public:
  DistributedWorkerGrpcServer(std::shared_ptr<DistributedModelLoader> servable, const std::string server_address)
      : GrpcAsyncServer<proto::MSDistributedWorker::AsyncService>(),
        service_impl_(MSDistributedImpl(servable, server_address)) {}

  void EnqueueRequests() override {
    WorkerAgentRegisterContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    WorkerAgentExitContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    WorkerAgentFailedContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    WorkerAgentConfigAcquireContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    WorkerPingContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    WorkerPongContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
  }

 private:
  MSDistributedImpl service_impl_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_DISTRIBUTED_WORKER_SERVER_H
