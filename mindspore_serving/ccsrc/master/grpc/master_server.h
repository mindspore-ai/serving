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

#ifndef MINDSPORE_SERVING_MASTER_MASTER_SERVER_H
#define MINDSPORE_SERVING_MASTER_MASTER_SERVER_H

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

template <class Derived>
class MasterGrpcContext : public GrpcAsyncServiceContext<MSMasterImpl, proto::MSMaster::AsyncService, Derived> {
 public:
  MasterGrpcContext(MSMasterImpl *service_impl, proto::MSMaster::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : GrpcAsyncServiceContext<MSMasterImpl, proto::MSMaster::AsyncService, Derived>(service_impl, async_service, cq) {
  }

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;
};

class MasterRegisterContext : public MasterGrpcContext<MasterRegisterContext> {
 public:
  MasterRegisterContext(MSMasterImpl *service_impl, proto::MSMaster::AsyncService *async_service,
                        grpc::ServerCompletionQueue *cq)
      : MasterGrpcContext<MasterRegisterContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~MasterRegisterContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestRegister(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->Register(&request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::RegisterReply> responder_;
  proto::RegisterRequest request_;
  proto::RegisterReply response_;
};

class MasterExitContext : public MasterGrpcContext<MasterExitContext> {
 public:
  MasterExitContext(MSMasterImpl *service_impl, proto::MSMaster::AsyncService *async_service,
                    grpc::ServerCompletionQueue *cq)
      : MasterGrpcContext<MasterExitContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~MasterExitContext() = default;

  void StartEnqueueRequest() override { async_service_->RequestExit(&ctx_, &request_, &responder_, cq_, cq_, this); }

  void HandleRequest() override {
    grpc::Status status = service_impl_->Exit(&request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::ExitReply> responder_;
  proto::ExitRequest request_;
  proto::ExitReply response_;
};

class MasterNotifyFailedContext : public MasterGrpcContext<MasterNotifyFailedContext> {
 public:
  MasterNotifyFailedContext(MSMasterImpl *service_impl, proto::MSMaster::AsyncService *async_service,
                            grpc::ServerCompletionQueue *cq)
      : MasterGrpcContext<MasterNotifyFailedContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~MasterNotifyFailedContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestNotifyFailed(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->NotifyFailed(&request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::NotifyFailedReply> responder_;
  proto::NotifyFailedRequest request_;
  proto::NotifyFailedReply response_;
};

class MasterGetModelInfoContext : public MasterGrpcContext<MasterGetModelInfoContext> {
 public:
  MasterGetModelInfoContext(MSMasterImpl *service_impl, proto::MSMaster::AsyncService *async_service,
                            grpc::ServerCompletionQueue *cq)
      : MasterGrpcContext<MasterGetModelInfoContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~MasterGetModelInfoContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestGetModelInfo(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    grpc::Status status = service_impl_->GetModelInfo(&request_, &response_);
    responder_.Finish(response_, status, this);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::GetModelInfoReply> responder_;
  proto::GetModelInfoRequest request_;
  proto::GetModelInfoReply response_;
};

class MasterPredictContext : public MasterGrpcContext<MasterPredictContext> {
 public:
  MasterPredictContext(MSMasterImpl *service_impl, proto::MSMaster::AsyncService *async_service,
                       grpc::ServerCompletionQueue *cq)
      : MasterGrpcContext<MasterPredictContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~MasterPredictContext() = default;

  void StartEnqueueRequest() override {
    async_service_->RequestCallModel(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void HandleRequest() override {
    PredictOnFinish on_finish = [this]() { responder_.Finish(response_, grpc::Status::OK, this); };
    service_impl_->PredictAsync(&request_, &response_, on_finish);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder_;
  proto::PredictRequest request_;
  proto::PredictReply response_;
};

class MasterGrpcServer : public GrpcAsyncServer<proto::MSMaster::AsyncService> {
 public:
  explicit MasterGrpcServer(std::shared_ptr<Dispatcher> dispatcher)
      : GrpcAsyncServer<proto::MSMaster::AsyncService>(), service_impl_(MSMasterImpl(dispatcher)) {}
  ~MasterGrpcServer() {}

  void EnqueueRequests() override {
    MasterRegisterContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    MasterExitContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    MasterNotifyFailedContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    MasterGetModelInfoContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
    MasterPredictContext::EnqueueRequest(&service_impl_, &svc_, cq_.get());
  }

 protected:
  MSMasterImpl service_impl_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_MASTER_SERVER_H
