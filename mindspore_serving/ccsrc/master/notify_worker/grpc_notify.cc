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
#include "master/notify_worker/grpc_notify.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <thread>
#include "common/exit_handle.h"
#include "common/grpc_server.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {

GrpcNotifyWorker::GrpcNotifyWorker(const std::string &worker_address) {
  worker_address_ = worker_address;
  std::shared_ptr<grpc::Channel> channel = GrpcServer::CreateChannel(worker_address);
  stub_ = proto::MSWorker::NewStub(channel);
}

GrpcNotifyWorker::~GrpcNotifyWorker() = default;

Status GrpcNotifyWorker::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                                       PredictOnFinish on_finish) {
  if (!stub_) {
    return INFER_STATUS_LOG_ERROR(WORKER_UNAVAILABLE)
           << "Predict failed, worker gRPC has not been inited or has already exited, worker address "
           << worker_address_;
  }
  if (!client_) {
    client_ = std::make_unique<MSPredictClient>();
    client_->Start();
  }
  AsyncPredictCallback callback = [reply, on_finish](Status status) {
    GrpcTensorHelper::CreateReplyFromErrorMsg(status, reply);
    on_finish();
  };
  client_->PredictAsync(request, reply, stub_.get(), callback, worker_address_);
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
