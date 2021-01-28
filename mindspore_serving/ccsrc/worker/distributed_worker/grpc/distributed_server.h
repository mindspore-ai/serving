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
#include "worker/distributed_worker/grpc/distributed_process.h"

namespace mindspore {
namespace serving {

// Service Implement
class MS_API MSDistributedWorkerServer : public MSWorkerServer {
 public:
  Status StartDistributedWorkerGrpcServer(std::shared_ptr<DistributedServable> servable, const std::string &hostname,
                                          int32_t port);
};

// Service Implement
class WorkerAgentRegisterContext : public WorkerServiceContext {
 public:
  WorkerAgentRegisterContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq), responder_(&ctx_) {
    state_ = STATE::CREATE;
  }

  ~WorkerAgentRegisterContext() = default;

  static Status EnqueueRequest(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerAgentRegisterContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestPredict(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = service_impl_->Predict(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

  bool JudgeFinish() override { return state_ == STATE::FINISH; }

 private:
  MSDistributedImpl *service_impl_;
  proto::MSWorker::AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder_;
  proto::PredictRequest request_;
  proto::PredictReply response_;
};

class WorkerAgentExitContext : public WorkerServiceContext {
 public:
  WorkerAgentExitContext(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                         grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq), responder_(&ctx_) {
    state_ = STATE::CREATE;
  }

  ~WorkerAgentExitContext() = default;

  static Status EnqueueRequest(MSDistributedImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerAgentExitContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestExit(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = service_impl_->Exit(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

  bool JudgeFinish() override { return state_ == STATE::FINISH; }

 private:
  MSDistributedImpl *service_impl_;
  proto::MSWorker::AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<proto::ExitReply> responder_;
  proto::ExitRequest request_;
  proto::ExitReply response_;
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
    return SUCCESS;
  }

 private:
  MSDistributedImpl *distributed_service_impl_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_DISTRIBUTED_WORKER_SERVER_H
