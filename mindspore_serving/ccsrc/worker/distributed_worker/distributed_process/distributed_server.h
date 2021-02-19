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

// Service Implement
class MS_API MSDistributedWorkerServer : public MSWorkerServer {
 public:
  explicit MSDistributedWorkerServer(std::shared_ptr<DistributedServable> servable) : servable_(servable) {}
  ~MSDistributedWorkerServer() = default;
  Status StartWorkerGrpcServer(const std::string &hostname, int32_t port) override;

 private:
  std::shared_ptr<DistributedServable> servable_;
};

class DistributedServiceContext : public WorkerServiceContext {
 public:
  DistributedServiceContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                            grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), dist_service_impl_(service_impl) {}

 protected:
  MSDistributedImpl *dist_service_impl_ = nullptr;
};

// Service Implement
class WorkerAgentRegisterContext : public DistributedServiceContext {
 public:
  WorkerAgentRegisterContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentRegisterContext() = default;

  static Status EnqueueRequest(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerAgentRegisterContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestAgentRegister(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(dist_service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = dist_service_impl_->AgentRegister(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentRegisterReply> responder_;
  proto::AgentRegisterRequest request_;
  proto::AgentRegisterReply response_;
};

class WorkerAgentExitContext : public DistributedServiceContext {
 public:
  WorkerAgentExitContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                         grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentExitContext() = default;

  static Status EnqueueRequest(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerAgentExitContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestAgentExit(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(dist_service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = dist_service_impl_->AgentExit(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentExitReply> responder_;
  proto::AgentExitRequest request_;
  proto::AgentExitReply response_;
};

class WorkerAgentFailedContext : public DistributedServiceContext {
 public:
  WorkerAgentFailedContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                           grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentFailedContext() = default;
  static Status EnqueueRequest(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerAgentFailedContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestAgentFailed(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(dist_service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = dist_service_impl_->AgentFailed(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentFailedReply> responder_;
  proto::AgentFailedRequest request_;
  proto::AgentFailedReply response_;
};

class WorkerAgentConfigAcquireContext : public DistributedServiceContext {
 public:
  WorkerAgentConfigAcquireContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                                  grpc::ServerCompletionQueue *cq)
      : DistributedServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerAgentConfigAcquireContext() = default;
  static Status EnqueueRequest(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerAgentConfigAcquireContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestAgentConfigAcquire(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(dist_service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = dist_service_impl_->AgentConfigAcquire(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::AgentConfigAcquireReply> responder_;
  proto::AgentConfigAcquireRequest request_;
  proto::AgentConfigAcquireReply response_;
};

class DistributedWorkerGrpcServer : public WorkerGrpcServer {
 public:
  DistributedWorkerGrpcServer(const std::string &host, int32_t port, MSDistributedImpl *service_impl)
      : WorkerGrpcServer(host, port, service_impl), distributed_service_impl_(service_impl) {}

  ~DistributedWorkerGrpcServer() = default;

  Status EnqueueRequest() {
    WorkerGrpcServer::EnqueueRequest();
    WorkerAgentRegisterContext::EnqueueRequest(distributed_service_impl_, &svc_, cq_.get());
    WorkerAgentExitContext::EnqueueRequest(distributed_service_impl_, &svc_, cq_.get());
    WorkerAgentFailedContext::EnqueueRequest(distributed_service_impl_, &svc_, cq_.get());
    WorkerAgentConfigAcquireContext::EnqueueRequest(distributed_service_impl_, &svc_, cq_.get());
    return SUCCESS;
  }

 private:
  MSDistributedImpl *distributed_service_impl_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_DISTRIBUTED_WORKER_SERVER_H
