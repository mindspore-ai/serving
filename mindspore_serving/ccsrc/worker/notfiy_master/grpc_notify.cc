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
#include "common/grpc_server.h"
#include "worker/servable_register.h"
#include "common/shared_memory.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {
GrpcNotifyMaster::GrpcNotifyMaster(const std::string &master_address, const std::string &worker_address)
    : master_address_(master_address), worker_address_(worker_address) {
  auto channel = GrpcServer::CreateChannel(master_address_);
  stub_ = proto::MSMaster::NewStub(channel);
}

GrpcNotifyMaster::~GrpcNotifyMaster() = default;

Status GrpcNotifyMaster::Register(const WorkerRegSpec &worker_spec) {
  proto::RegisterRequest request;
  GrpcTensorHelper::ConvertWorkerSpec(worker_spec, &request);

  MSI_LOG(INFO) << "Register to " << master_address_;
  proto::RegisterReply reply;
  grpc::ClientContext context;
  const int32_t TIME_OUT = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
  context.set_deadline(deadline);
  grpc::Status status = stub_->Register(&context, request, &reply);
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
  grpc::Status status = stub_->Exit(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Exit SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_WARNING(FAILED)
         << "Exit Failed, master may have exited, Grpc message: " << status.error_code() << ", "
         << status.error_message();
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

Status GrpcNotifyMaster::GetModelInfos(const std::string &master_address, const std::string &servable_name,
                                       uint32_t version_number, proto::GetModelInfoReply *reply) {
  proto::GetModelInfoRequest request;
  request.set_servable_name(servable_name);
  request.set_version_number(version_number);
  auto channel = GrpcServer::CreateChannel(master_address);
  auto stub = proto::MSMaster::NewStub(channel);

  grpc::ClientContext context;
  grpc::Status grpc_status = stub->GetModelInfo(&context, request, reply);
  if (!grpc_status.ok()) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
           << "Get model infos failed, master address:" << master_address
           << ", Grpc message: " << grpc_status.error_code() << ", " << grpc_status.error_message();
  }
  return SUCCESS;
}

Status GrpcNotifyMaster::CreateRequestShmInstance(const RemoteCallModelContext &model_context,
                                                  const InstanceData &instance, proto::Instance *proto_instance,
                                                  std::vector<SharedMemoryItem> *alloc_shm_request) {
  Status status;
  auto &memory_instance = SharedMemoryAllocator::Instance();
  auto &proto_items = *(proto_instance->mutable_items());
  for (size_t i = 0; i < instance.size(); i++) {
    auto &input = instance[i];
    auto &memory_key = model_context.request_memory[i];
    SharedMemoryItem memory_item;
    status = memory_instance.AllocMemoryItem(memory_key, &memory_item);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Alloc request memory failed, memory: " << memory_key;
      return status;
    }
    alloc_shm_request->push_back(memory_item);
    auto &proto_tensor = proto_items["x" + std::to_string(i)];  // input: x0, x1, x2,...
    ProtoTensor tensor(&proto_tensor);
    tensor.set_data_type(input->data_type());
    tensor.set_shape(input->shape());
    auto proto_shm_data = proto_tensor.mutable_shm_data();
    proto_shm_data->set_memory_key(memory_item.memory_key);
    proto_shm_data->set_bytes_size(memory_item.bytes_size);
    proto_shm_data->set_data_size(memory_item.size);
    proto_shm_data->set_data_offset(memory_item.offset);
    auto ret = memcpy_s(memory_item.offset_address, memory_item.size, input->data(), input->data_size());
    if (ret != EOK) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Copy tensor to shared memory failed, dst size: " << memory_item.size
                                            << ", src size: " << input->data_size();
    }
  }
  return SUCCESS;
}

Status GrpcNotifyMaster::CreateResultShmInstance(const RemoteCallModelContext &model_context,
                                                 ResultInstance *result_instance, proto::Instance *proto_instance) {
  Status status;
  auto &memory_instance = SharedMemoryAllocator::Instance();
  auto &proto_reply_items = *(proto_instance->mutable_output_buffers());
  for (size_t i = 0; i < model_context.output_infos.size(); i++) {
    auto &output_info = model_context.output_infos[i];
    auto &memory_key = model_context.reply_memory[i];
    SharedMemoryItem memory_item;
    status = memory_instance.AllocMemoryItem(memory_key, &memory_item);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Alloc request memory failed, memory: " << memory_key;
      return status;
    }
    auto &proto_output = proto_reply_items["y" + std::to_string(i)];
    proto_output.set_memory_key(memory_item.memory_key);
    proto_output.set_bytes_size(memory_item.bytes_size);
    proto_output.set_data_size(memory_item.size);
    proto_output.set_data_offset(memory_item.offset);
    auto result_tensor =
      std::make_shared<ShmTensor>(output_info.tensor_info.data_type, output_info.shape_one_batch, memory_item);
    result_instance->data.push_back(result_tensor);
  }
  return SUCCESS;
}

Status GrpcNotifyMaster::CallModelInner(const RemoteCallModelContext &model_context,
                                        const std::vector<InstanceData> &request, std::vector<ResultInstance> *reply,
                                        std::vector<SharedMemoryItem> *alloc_shm_request) {
  proto::PredictRequest proto_request;
  auto servable_spec = proto_request.mutable_servable_spec();
  servable_spec->set_name(ServableRegister::Instance().GetServableSignature().servable_name);
  servable_spec->set_method_name(
    ServableRegister::GetCallModelMethodName(model_context.model_name, model_context.subgraph));
  servable_spec->set_version_number(model_context.version_number);
  auto proto_instances = proto_request.mutable_instances();
  Status status;
  std::vector<ResultInstance> result_instances;
  for (auto &instance : request) {
    auto proto_instance = proto_instances->Add();
    status = CreateRequestShmInstance(model_context, instance, proto_instance, alloc_shm_request);
    if (status != SUCCESS) {
      return status;
    }
    ResultInstance result_instance;
    status = CreateResultShmInstance(model_context, &result_instance, proto_instance);
    if (status != SUCCESS) {
      return status;
    }
    result_instances.push_back(result_instance);
  }
  proto::PredictReply proto_reply;
  MSI_TIME_STAMP_START(CallModel)
  grpc::ClientContext context;
  grpc::Status grpc_status = stub_->CallModel(&context, proto_request, &proto_reply);
  MSI_TIME_STAMP_END_EXTRA(CallModel, "Request count " + std::to_string(request.size()))
  if (!grpc_status.ok()) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
           << "Remote call model failed, master address:" << master_address_
           << ", Grpc message: " << grpc_status.error_code() << ", " << grpc_status.error_message();
  }
  auto &error_msgs = proto_reply.error_msg();
  auto &reply_instances = proto_reply.instances();
  if (error_msgs.size() == 1 && error_msgs[0].error_code() != 0) {
    if (error_msgs[0].error_code() == SERVABLE_UNAVAILABLE) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "There are no available inference processes that occupy devices";
    }
    return INFER_STATUS_LOG_ERROR(FAILED) << "Remote call model failed: " << error_msgs[0].error_msg();
  }
  if (!reply_instances.empty() && static_cast<size_t>(reply_instances.size()) != request.size()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Remote call model failed, reply instances size " << reply_instances.size()
                                          << " is not equal to request instances size " << request.size();
  }
  for (int i = 0; i < reply_instances.size(); i++) {
    ResultInstance result_instance;
    if (i < error_msgs.size() && error_msgs[i].error_code() != 0) {
      result_instance.error_msg = INFER_STATUS_LOG_ERROR(FAILED)
                                  << "Result instance " << i << "failed: " << error_msgs[i].error_msg();
    } else {
      auto &proto_instance = reply_instances[i];
      auto &proto_items = proto_instance.items();
      for (auto &output : proto_items) {
        if (!output.second.has_shm_data()) {
          return INFER_STATUS_LOG_ERROR(FAILED) << "Result instance " << i << " invalid, there no shared memory data";
        }
      }
      result_instance.data = result_instances[i].data;
    }
    reply->push_back(result_instance);
  }
  return SUCCESS;
}

Status GrpcNotifyMaster::CallModel(const RemoteCallModelContext &model_context,
                                   const std::vector<InstanceData> &request, std::vector<ResultInstance> *reply) {
  std::vector<SharedMemoryItem> alloc_shm_request;
  auto status = CallModelInner(model_context, request, reply, &alloc_shm_request);
  auto &memory_instance = SharedMemoryAllocator::Instance();
  for (auto &item : alloc_shm_request) {
    memory_instance.ReleaseMemoryItem(item);
  }
  return status;
}
}  // namespace serving
}  // namespace mindspore
