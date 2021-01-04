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
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <thread>
#include "common/exit_handle.h"
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {

GrpcNotfiyWorker::GrpcNotfiyWorker(const std::string &worker_address) {
  worker_address_ = worker_address;
  std::shared_ptr<grpc::Channel> channel = GrpcServer::CreateChannel(worker_address);
  stub_ = proto::MSWorker::NewStub(channel);
}

GrpcNotfiyWorker::~GrpcNotfiyWorker() = default;

Status GrpcNotfiyWorker::Dispatch(const proto::PredictRequest &request, proto::PredictReply *reply) {
  if (!stub_) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Predict failed, worker gRPC has not been inited or has already exited, worker address "
           << worker_address_;
  }
  grpc::ClientContext context;
  auto status = stub_->Predict(&context, request, reply);
  if (!status.ok()) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Predict failed, worker gRPC error: " << status.error_code() << ", " << status.error_message();
  }
  return SUCCESS;
}

Status GrpcNotfiyWorker::Exit() {
  if (stub_) {
    proto::ExitRequest request;
    request.set_address(worker_address_);
    proto::ExitReply reply;
    grpc::ClientContext context;
    const int32_t TIME_OUT = 1;
    std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
    context.set_deadline(deadline);

    (void)stub_->Exit(&context, request, &reply);
  }
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
