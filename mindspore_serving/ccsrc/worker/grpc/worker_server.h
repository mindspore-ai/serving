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
#include "worker/distributed_worker/distributed_servable.h"

namespace mindspore {
namespace serving {

// Service Implement
class MS_API MSWorkerServer {
 public:
  enum ServerState { kGdsUninit = 0, kGdsInitializing, kGdsRunning, kGdsStopped };
  MSWorkerServer();
  virtual ~MSWorkerServer();

  virtual Status StartWorkerGrpcServer(const std::string &hostname, int32_t port);
  Status Stop();

 protected:
  bool in_running_ = false;
  std::thread grpc_thread_;
  std::unique_ptr<MSWorkerImpl> service_impl_ = nullptr;
  std::unique_ptr<GrpcAsyncServer> async_server_ = nullptr;

  Status Init();
  Status StartAsyncRpcService();
};

class WorkerServiceContext {
 public:
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };

  WorkerServiceContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq) {
    state_ = STATE::CREATE;
  }
  virtual ~WorkerServiceContext() {}

  bool JudgeFinish() { return state_ == STATE::FINISH; }

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;

 protected:
  MSWorkerImpl *service_impl_;
  proto::MSWorker::AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  grpc::ServerContext ctx_;

  STATE state_;
};

class WorkerPredictContext : public WorkerServiceContext {
 public:
  WorkerPredictContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

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

class WorkerExitContext : public WorkerServiceContext {
 public:
  WorkerExitContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

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

 private:
  grpc::ServerAsyncResponseWriter<proto::ExitReply> responder_;
  proto::ExitRequest request_;
  proto::ExitReply response_;
};

class WorkerPingContext : public WorkerServiceContext {
 public:
  WorkerPingContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerPingContext() = default;

  static Status EnqueueRequest(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerPingContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestPing(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = service_impl_->Ping(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PingReply> responder_;
  proto::PingRequest request_;
  proto::PingReply response_;
};

class WorkerPongContext : public WorkerServiceContext {
 public:
  WorkerPongContext(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : WorkerServiceContext(service_impl, async_service, cq), responder_(&ctx_) {}

  ~WorkerPongContext() = default;

  static Status EnqueueRequest(MSWorkerImpl *service_impl, proto::MSWorker::AsyncService *async_service,
                               grpc::ServerCompletionQueue *cq) {
    auto call = new WorkerPongContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
    return SUCCESS;
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestPong(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;
    grpc::Status status = service_impl_->Pong(&ctx_, &request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PongReply> responder_;
  proto::PongRequest request_;
  proto::PongReply response_;
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
    WorkerPingContext::EnqueueRequest(service_impl_, &svc_, cq_.get());
    WorkerPongContext::EnqueueRequest(service_impl_, &svc_, cq_.get());
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

 protected:
  MSWorkerImpl *service_impl_;
  proto::MSWorker::AsyncService svc_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_PROCESS_H
