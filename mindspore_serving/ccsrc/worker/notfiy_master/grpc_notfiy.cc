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
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <thread>
#include "common/exit_handle.h"

namespace mindspore {
namespace serving {

GrpcNotfiyMaster::GrpcNotfiyMaster(const std::string &master_ip, uint32_t master_port, const std::string &host_ip,
                                   uint32_t host_port)
    : master_ip_(master_ip), master_port_(master_port), host_ip_(host_ip), host_port_(host_port) {
  master_address_ = master_ip_ + ":" + std::to_string(master_port);
  worker_address_ = host_ip_ + ":" + std::to_string(host_port_);
  auto channel = grpc::CreateChannel(master_address_, grpc::InsecureChannelCredentials());
  stub_ = proto::MSMaster::NewStub(channel);
}

GrpcNotfiyMaster::~GrpcNotfiyMaster() = default;

Status GrpcNotfiyMaster::Register(const std::vector<WorkerSpec> &worker_specs) {
  const int32_t REGISTER_TIME_OUT = 60;
  const int32_t REGISTER_INTERVAL = 1;
  auto loop = REGISTER_TIME_OUT;
  while (loop-- && !ExitHandle::Instance().HasStopped()) {
    MSI_LOG(INFO) << "Register to " << master_address_;
    proto::RegisterRequest request;
    request.set_address(worker_address_);
    for (size_t i = 0; i < worker_specs.size(); i++) {
      auto &spec = worker_specs[i];
      auto worker_spec = request.add_worker_spec();
      worker_spec->set_name(spec.servable_name);
      worker_spec->set_version_number(spec.version_number);
      for (auto &method : spec.methods) {
        auto proto_method = worker_spec->add_methods();
        proto_method->set_name(method.name);
        for (auto &name : method.input_names) {
          proto_method->add_input_names(name);
        }
      }
    }

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
    std::this_thread::sleep_for(std::chrono::milliseconds(REGISTER_INTERVAL * 1000));
  }
  if (ExitHandle::Instance().HasStopped()) {
    return INFER_STATUS_LOG_WARNING(FAILED) << "Worker exit, stop registration";
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Register TimeOut";
}

Status GrpcNotfiyMaster::Unregister() {
  if (is_stoped_.load()) {
    return SUCCESS;
  }
  is_stoped_ = true;
  proto::ExitRequest request;
  request.set_address(worker_address_);
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

Status GrpcNotfiyMaster::AddWorker(const WorkerSpec &worker_spec) {
  proto::AddWorkerReply reply;
  grpc::ClientContext context;
  proto::AddWorkerRequest request;
  request.set_address(worker_address_);
  request.mutable_worker_spec()->set_name(worker_spec.servable_name);
  request.mutable_worker_spec()->set_version_number(worker_spec.version_number);
  for (auto &method : worker_spec.methods) {
    auto proto_method = request.mutable_worker_spec()->add_methods();
    proto_method->set_name(method.name);
    for (auto &name : method.input_names) {
      proto_method->add_input_names(name);
    }
  }
  const int32_t INTERVAL = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(INTERVAL);
  context.set_deadline(deadline);
  grpc::Status status = stub_->AddWorker(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "AddWorker SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "AddWorker Failed";
}

Status GrpcNotfiyMaster::RemoveWorker(const WorkerSpec &worker_spec) {
  proto::RemoveWorkerReply reply;
  grpc::ClientContext context;
  proto::RemoveWorkerRequest request;
  request.set_address(worker_address_);
  request.mutable_worker_spec()->set_name(worker_spec.servable_name);
  request.mutable_worker_spec()->set_version_number(worker_spec.version_number);
  for (auto &method : worker_spec.methods) {
    auto proto_method = request.mutable_worker_spec()->add_methods();
    proto_method->set_name(method.name);
    for (auto &name : method.input_names) {
      proto_method->add_input_names(name);
    }
  }
  const int32_t INTERVAL = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(INTERVAL);
  context.set_deadline(deadline);
  grpc::Status status = stub_->RemoveWorker(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "RemoveWorker SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "RemoveWorker Failed";
}

}  // namespace serving
}  // namespace mindspore
