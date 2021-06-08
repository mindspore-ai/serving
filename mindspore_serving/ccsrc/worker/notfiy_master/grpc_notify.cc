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
#include "worker/notfiy_master/grpc_notify.h"
#include <unistd.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <thread>
#include "common/exit_handle.h"
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {

GrpcNotifyMaster::GrpcNotifyMaster(const std::string &master_address, const std::string &worker_address)
    : master_address_(master_address), worker_address_(worker_address) {}

GrpcNotifyMaster::~GrpcNotifyMaster() = default;

Status GrpcNotifyMaster::Register(const WorkerRegSpec &worker_spec) {
  proto::RegisterRequest request;
  auto proto_worker_spec = request.mutable_worker_spec();
  proto_worker_spec->set_address(worker_address_);
  proto_worker_spec->set_worker_pid(getpid());
  const auto &spec = worker_spec.servable_spec;
  auto proto_spec = proto_worker_spec->mutable_servable_spec();
  proto_spec->set_name(spec.servable_name);
  proto_spec->set_version_number(spec.version_number);
  proto_spec->set_batch_size(spec.batch_size);
  for (auto &method : spec.methods) {
    auto proto_method = proto_spec->add_methods();
    proto_method->set_name(method.name);
    for (auto &name : method.input_names) {
      proto_method->add_input_names(name);
    }
  }

  MSI_LOG(INFO) << "Register to " << master_address_;
  proto::RegisterReply reply;
  grpc::ClientContext context;
  const int32_t TIME_OUT = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
  context.set_deadline(deadline);
  auto channel = GrpcServer::CreateChannel(master_address_);
  auto stub = proto::MSMaster::NewStub(channel);
  grpc::Status status = stub->Register(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Register SUCCESS ";
    is_running_ = true;
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
         << "Register failed, Grpc message: " << status.error_code() << ", " << status.error_message();
}

Status GrpcNotifyMaster::Unregister() {
  if (!is_running_) {
    return SUCCESS;
  }
  is_running_ = false;
  proto::ExitRequest request;
  request.set_address(worker_address_);
  MSI_LOG(INFO) << "Unregister to " << master_address_;
  proto::ExitReply reply;
  grpc::ClientContext context;
  const int32_t TIME_OUT = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
  context.set_deadline(deadline);
  auto channel = GrpcServer::CreateChannel(master_address_);
  auto stub = proto::MSMaster::NewStub(channel);
  grpc::Status status = stub->Exit(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Exit SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
         << "Exit Failed, Grpc message: " << status.error_code() << ", " << status.error_message();
}

Status GrpcNotifyMaster::NotifyFailed(const std::string &master_address, const std::string &error_msg) {
  proto::NotifyFailedRequest request;
  request.set_worker_pid(getpid());
  request.set_error_msg(error_msg);
  auto channel = GrpcServer::CreateChannel(master_address);
  auto stub = proto::MSMaster::NewStub(channel);

  proto::NotifyFailedReply reply;
  grpc::ClientContext context;
  grpc::Status status = stub->NotifyFailed(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Success to notify master " << master_address << " error message of worker: " << error_msg;
    return SUCCESS;
  }
  MSI_LOG_WARNING << "Failed to notify master " << master_address << " error message of worker: " << error_msg
                  << ", grpc error: " << status.error_message();
  return FAILED;
}

}  // namespace serving
}  // namespace mindspore
