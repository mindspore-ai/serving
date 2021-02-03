/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "worker/distributed_worker/notify_agent/notify_agent.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <thread>
#include "common/exit_handle.h"
#include "common/grpc_server.h"
#include "common/grpc_client.h"

namespace mindspore {
namespace serving {

GrpcNotifyAgent::GrpcNotifyAgent(const std::string &agent_address) {
  agent_address_ = agent_address;
  std::shared_ptr<grpc::Channel> channel = GrpcServer::CreateChannel(agent_address_);
  stub_ = proto::MSAgent::NewStub(channel);
}

GrpcNotifyAgent::~GrpcNotifyAgent() = default;

Status GrpcNotifyAgent::Exit() {
  MSI_LOG_INFO << "Notify one agent exit begin";
  if (stub_) {
    proto::DistributedExitRequest request;
    request.set_address(agent_address_);
    proto::DistributedExitReply reply;
    grpc::ClientContext context;
    const int32_t TIME_OUT = 1;
    std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
    context.set_deadline(deadline);

    auto status = stub_->Exit(&context, request, &reply);
    if (status.ok()) {
      MSI_LOG_INFO << "Notify one agent exit success, agent address: " << agent_address_;
    } else {
      MSI_LOG_INFO << "Notify one agent exit failed, agent address: " << agent_address_
                   << ", error: " << status.error_code() << ", " << status.error_message();
    }
  }
  return SUCCESS;
}

Status GrpcNotifyAgent::DispatchAsync(const proto::DistributedPredictRequest &request,
                                      proto::DistributedPredictReply *reply, DispatchCallback callback) {
  if (!stub_) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Predict failed, agent gRPC has not been inited or has already exited, agent address " << agent_address_;
  }
  if (!distributed_client_) {
    distributed_client_ = std::make_unique<MSDistributedClient>();
    distributed_client_->Start();
  }
  distributed_client_->PredictAsync(request, reply, stub_.get(), callback);
  return SUCCESS;
}  // namespace serving

}  // namespace serving
}  // namespace mindspore
