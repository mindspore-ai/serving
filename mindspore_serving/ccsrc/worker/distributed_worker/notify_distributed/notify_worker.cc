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
#include "worker/distributed_worker/notify_distributed/notify_worker.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <thread>
#include "common/exit_handle.h"
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {

GrpcNotfiyDistributeWorker::GrpcNotfiyDistributeWorker(const std::string &distributed_worker_ip,
                                                       uint32_t distributed_worker_port, const std::string &host_ip,
                                                       uint32_t host_port)
    : distributed_worker_ip_(distributed_worker_ip),
      distributed_worker_port_(distributed_worker_port),
      host_ip_(host_ip),
      host_port_(host_port) {
  distributed_worker_address_ = distributed_worker_ip + ":" + std::to_string(distributed_worker_port);
  agent_address_ = host_ip_ + ":" + std::to_string(host_port_);
  auto channel = GrpcServer::CreateChannel(distributed_worker_address_);
  stub_ = proto::MSDistributedWorker::NewStub(channel);
}

GrpcNotfiyDistributeWorker::~GrpcNotfiyDistributeWorker() = default;

Status GrpcNotfiyDistributeWorker::Register(const std::vector<WorkerAgentSpec> &worker_specs) {
  const int32_t REGISTER_TIME_OUT = 60;
  const int32_t REGISTER_INTERVAL = 1;
  auto loop = REGISTER_TIME_OUT;
  while (loop-- && !ExitSignalHandle::Instance().HasStopped()) {
    MSI_LOG(INFO) << "Register to " << distributed_worker_address_;
    proto::RegisterRequest request;
    request.set_address(agent_address_);
    // to do set RegisterRequest message
    proto::RegisterReply reply;
    grpc::ClientContext context;
    std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(REGISTER_INTERVAL);
    context.set_deadline(deadline);
    grpc::Status status = stub_->Register(&context, request, &reply);
    if (status.ok()) {
      MSI_LOG(INFO) << "Register SUCCESS ";
      return SUCCESS;
    }
    MSI_LOG_INFO << "Grpc message: " << status.error_code() << ", " << status.error_message();
    std::this_thread::sleep_for(std::chrono::milliseconds(REGISTER_INTERVAL * 1000));
  }
  if (ExitSignalHandle::Instance().HasStopped()) {
    return INFER_STATUS_LOG_WARNING(FAILED) << "Worker exit, stop registration";
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Register TimeOut";
}

Status GrpcNotfiyDistributeWorker::Unregister() {
  if (is_stoped_.load()) {
    return SUCCESS;
  }
  is_stoped_ = true;
  proto::ExitRequest request;
  request.set_address(agent_address_);
  proto::ExitReply reply;
  grpc::ClientContext context;
  const int32_t TIME_OUT = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
  context.set_deadline(deadline);
  grpc::Status status = stub_->Exit(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Exit SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Exit Failed";
}

}  // namespace serving
}  // namespace mindspore
