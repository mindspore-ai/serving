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

#ifndef MINDSPORE_SERVING_MASTER_GRPC_SERVER_H
#define MINDSPORE_SERVING_MASTER_GRPC_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <string>
#include <vector>
#include <memory>
#include "common/serving_common.h"
#include "proto/ms_worker.pb.h"
#include "proto/ms_worker.grpc.pb.h"
#include "common/grpc_async_server.h"
#include "master/grpc/grpc_process.h"

namespace mindspore {
namespace serving {

class MSServiceServer;
extern std::unique_ptr<MSServiceServer> grpc_async_server_;

// Service Implement
class MSServiceServer {
 public:
  enum ServerState { kGdsUninit = 0, kGdsInitializing, kGdsRunning, kGdsStopped };
  MSServiceServer(std::shared_ptr<MSServiceImpl> service, const std::string &hostname, int32_t port);
  ~MSServiceServer();

  Status Init(int max_msg_mb_size);

  Status Stop();

  Status StartAsyncRpcService();

  Status SendFinish();
  bool in_running_ = false;
  std::thread grpc_thread_;
  std::shared_ptr<MSServiceImpl> service_impl_;
  std::unique_ptr<GrpcAsyncServer> async_server_;
};

class MasterServiceContext {
 public:
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };

  virtual ~MasterServiceContext() {}

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;

  virtual bool JudgeFinish() = 0;

 public:
  STATE state_;
};

class MasterPredictContext : public MasterServiceContext {
 public:
  MasterPredictContext(MSServiceImpl *service_impl, proto::MSService::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq), responder_(&ctx_) {
    state_ = STATE::CREATE;
  }

  ~MasterPredictContext() = default;

  static void EnqueueRequest(MSServiceImpl *service_impl, proto::MSService::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq) {
    auto call = new MasterPredictContext(service_impl, async_service, cq);
    call->StartEnqueueRequest();
  }

  void StartEnqueueRequest() override {
    state_ = STATE::PROCESS;
    async_service_->RequestPredict(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    EnqueueRequest(service_impl_, async_service_, cq_);
    state_ = STATE::FINISH;

    MSI_TIME_STAMP_START(RequestHandle)
    PredictOnFinish on_finish = [this, time_start_RequestHandle]() {
      responder_.Finish(response_, grpc::Status::OK, this);
      MSI_TIME_STAMP_END(RequestHandle)
    };
    service_impl_->PredictAsync(&request_, &response_, on_finish);
  }

  bool JudgeFinish() override { return state_ == STATE::FINISH; }

 private:
  MSServiceImpl *service_impl_;
  proto::MSService::AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder_;
  proto::PredictRequest request_;
  proto::PredictReply response_;
};

class MasterGrpcServer : public GrpcAsyncServer {
 public:
  MasterGrpcServer(const std::string &host, int32_t port, MSServiceImpl *service_impl)
      : GrpcAsyncServer(host, port), service_impl_(service_impl) {}

  ~MasterGrpcServer() = default;

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return SUCCESS;
  }

  Status EnqueueRequest() {
    MasterPredictContext::EnqueueRequest(service_impl_, &svc_, cq_.get());
    return SUCCESS;
  }

  Status ProcessRequest(void *tag) {
    auto rq = static_cast<MasterServiceContext *>(tag);
    if (rq->JudgeFinish()) {  // End Finish
      delete rq;
    } else {  // Get new Request
      rq->HandleRequest();
    }
    return SUCCESS;
  }

 private:
  MSServiceImpl *service_impl_;
  proto::MSService::AsyncService svc_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_SERVER_H
