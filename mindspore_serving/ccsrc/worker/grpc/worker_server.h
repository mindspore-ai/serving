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

namespace mindspore {
namespace serving {

// Service Implement
class MSWorkerServer {
 public:
  enum ServerState { kGdsUninit = 0, kGdsInitializing, kGdsRunning, kGdsStopped };
  MSWorkerServer(const std::string &hostname, int32_t port);
  ~MSWorkerServer();

  Status Init();

  Status Stop();

  Status StartAsyncRpcService();

  bool in_running_ = false;
  std::thread grpc_thread_;
  std::unique_ptr<MSWorkerImpl> service_impl_;
  std::unique_ptr<GrpcAsyncServer> async_server_;
};

class WorkerServiceContext {
 public:
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };
  virtual ~WorkerServiceContext() {}

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;

  virtual bool JudgeFinish() = 0;

 public:
  STATE state_;
};

class WorkerPredictContext : public WorkerServiceContext {
 public:
  WorkerPredictContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq), responder_(&ctx_) {
    state_ = STATE::CREATE;
  }

  ~WorkerPredictContext() = default;

  static Status EnqueueRequest(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerPredictContext(service_impl, async_service, cq);
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
  MSWorkerImpl *service_impl_;
  proto::MSWorker::AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder_;
  proto::PredictRequest request_;
  proto::PredictReply response_;
};

class WorkerExitContext : public WorkerServiceContext {
 public:
  WorkerExitContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq), responder_(&ctx_) {
    state_ = STATE::CREATE;
  }

  ~WorkerExitContext() = default;

  static Status EnqueueRequest(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerExitContext(service_impl, async_service, cq);
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
  MSWorkerImpl *service_impl_;
  proto::MSWorker::AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<proto::ExitReply> responder_;
  proto::ExitRequest request_;
  proto::ExitReply response_;
};

class WorkerGrpcServer : public GrpcAsyncServer {
 public:
  WorkerGrpcServer(const std::string &host, int32_t port, MSWorkerImpl *service_impl)
      : GrpcAsyncServer(host, port), service_impl_(service_impl) {}

  ~WorkerGrpcServer() = default;

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return SUCCESS;
  }

  Status EnqueueRequest() {
    WorkerPredictContext::EnqueueRequest(service_impl_, &svc_, cq_.get());
    WorkerExitContext::EnqueueRequest(service_impl_, &svc_, cq_.get());
    return SUCCESS;
  }

  Status ProcessRequest(void *tag) {
    auto rq = static_cast<WorkerServiceContext *>(tag);
    if (rq->JudgeFinish()) {
      delete rq;
    } else {
      rq->HandleRequest();
    }
    return SUCCESS;
  }

 private:
  MSWorkerImpl *service_impl_;
  proto::MSWorker::AsyncService svc_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H
