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
class ServiceGrpcContext : public GrpcAsyncServiceContext<MSServiceImpl, proto::MSService::AsyncService, Derived> {
 public:
  ServiceGrpcContext(MSServiceImpl *service_impl, proto::MSService::AsyncService *async_service,
                     grpc::ServerCompletionQueue *cq)
      : GrpcAsyncServiceContext<MSServiceImpl, proto::MSService::AsyncService, Derived>(service_impl, async_service,
                                                                                        cq) {}

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;
};

class ServicePredictContext : public ServiceGrpcContext<ServicePredictContext> {
 public:
  ServicePredictContext(MSServiceImpl *service_impl, proto::MSService::AsyncService *async_service,
                        grpc::ServerCompletionQueue *cq)
      : ServiceGrpcContext<ServicePredictContext>(service_impl, async_service, cq), responder_(&ctx_) {}

  ~ServicePredictContext() = default;

  void StartEnqueueRequest() override { async_service_->RequestPredict(&ctx_, &request_, &responder_, cq_, cq_, this); }

  void HandleRequest() override {
    MSI_TIME_STAMP_START(RequestHandle)
    auto instance_size = request_.instances_size();
    PredictOnFinish on_finish = [this, time_start_RequestHandle, instance_size]() {
      responder_.Finish(response_, grpc::Status::OK, this);
      MSI_TIME_STAMP_END_EXTRA(RequestHandle, "Request count " + std::to_string(instance_size))
    };
    service_impl_->PredictAsync(&request_, &response_, on_finish);
  }

 private:
  grpc::ServerAsyncResponseWriter<proto::PredictReply> responder_;
  proto::PredictRequest request_;
  proto::PredictReply response_;
};

class ServiceGrpcServer : public GrpcAsyncServer<proto::MSService::AsyncService> {
 public:
  explicit ServiceGrpcServer(std::shared_ptr<Dispatcher> dispatcher)
      : GrpcAsyncServer<proto::MSService::AsyncService>(), service_impl_(MSServiceImpl(dispatcher)) {}
  ~ServiceGrpcServer() {}

  void EnqueueRequests() override { ServicePredictContext::EnqueueRequest(&service_impl_, &svc_, cq_.get()); }

 protected:
  MSServiceImpl service_impl_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_MASTER_GRPC_SERVER_H
