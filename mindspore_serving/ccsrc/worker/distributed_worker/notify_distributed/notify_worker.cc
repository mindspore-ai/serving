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
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {

GrpcNotifyDistributeWorker::GrpcNotifyDistributeWorker(const std::string &distributed_address,
                                                       const std::string &agent_address)
    : distributed_address_(distributed_address), agent_address_(agent_address) {
  auto channel = GrpcServer::CreateChannel(distributed_address_);
  stub_ = proto::MSDistributedWorker::NewStub(channel);
}

GrpcNotifyDistributeWorker::~GrpcNotifyDistributeWorker() = default;

Status GrpcNotifyDistributeWorker::Register(const std::vector<WorkerAgentSpec> &worker_specs) {
  const int32_t REGISTER_INTERVAL = 1;

  MSI_LOG(INFO) << "Register to worker " << distributed_address_ << ", agent address: " << agent_address_;
  proto::AgentRegisterRequest request;
  GrpcTensorHelper::CopyFromWorkerAgentSpec(worker_specs, &request);
  request.set_address(agent_address_);
  proto::AgentRegisterReply reply;
  grpc::ClientContext context;
  std::chrono::system_clock::time_point deadline =
    std::chrono::system_clock::now() + std::chrono::seconds(REGISTER_INTERVAL);
  context.set_deadline(deadline);
  grpc::Status status = stub_->AgentRegister(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Register SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
         << "Register to worker failed, grpc error: " << status.error_code() << ", " << status.error_message();
}

Status GrpcNotifyDistributeWorker::Unregister() {
  if (is_stoped_.load()) {
    return SUCCESS;
  }
  is_stoped_ = true;
  proto::AgentExitRequest request;
  request.set_address(agent_address_);
  proto::AgentExitReply reply;
  grpc::ClientContext context;
  const int32_t TIME_OUT = 1;
  std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
  context.set_deadline(deadline);
  grpc::Status status = stub_->AgentExit(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Exit SUCCESS ";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Exit Failed";
}

Status GrpcNotifyDistributeWorker::NotifyFailed(const std::string &distributed_address) {
  auto channel = GrpcServer::CreateChannel(distributed_address);
  auto stub = proto::MSDistributedWorker::NewStub(channel);

  grpc::ClientContext context;
  proto::AgentFailedRequest request;
  proto::AgentFailedReply reply;
  grpc::Status status = stub->AgentFailed(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Success to notify failure of agent";
    return SUCCESS;
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Failed to notify failure of agent";
}

void GrpcNotifyDistributeWorker::StartupNotifyExit(const std::string &distributed_address,
                                                   const std::string &agent_ip) {
  auto channel = GrpcServer::CreateChannel(distributed_address);
  auto stub = proto::MSDistributedWorker::NewStub(channel);

  grpc::ClientContext context;
  proto::AgentExitRequest request;
  request.set_agent_ip(agent_ip);
  proto::AgentExitReply reply;
  grpc::Status status = stub->AgentExit(&context, request, &reply);
  if (status.ok()) {
    MSI_LOG(INFO) << "Success to notify exit of agent start up process";
  } else {
    MSI_LOG(INFO) << "Failed to notify exit of agent start up process";
  }
}

Status GrpcNotifyDistributeWorker::GetAgentsConfigsFromWorker(const std::string &distributed_address,
                                                              DistributedServableConfig *config) {
  const int32_t REGISTER_TIME_OUT = 60;
  const int32_t REGISTER_INTERVAL = 1;
  auto loop = REGISTER_TIME_OUT;
  while (loop-- && !ExitSignalHandle::Instance().HasStopped()) {
    auto channel = GrpcServer::CreateChannel(distributed_address);
    auto stub = proto::MSDistributedWorker::NewStub(channel);

    grpc::ClientContext context;
    proto::AgentConfigAcquireRequest request;
    proto::AgentConfigAcquireReply reply;
    std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(REGISTER_INTERVAL);
    context.set_deadline(deadline);
    grpc::Status status = stub->AgentConfigAcquire(&context, request, &reply);
    if (status.ok()) {
      return ParseAgentConfigAcquireReply(reply, config);
    }
    MSI_LOG_INFO << "Grpc message: " << status.error_code() << ", " << status.error_message();
    std::this_thread::sleep_for(std::chrono::milliseconds(REGISTER_INTERVAL * 1000));
  }
  if (ExitSignalHandle::Instance().HasStopped()) {
    return INFER_STATUS_LOG_WARNING(FAILED) << "Agent exit, stop get Agents configs from Worker";
  }
  return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Failed to get Agents configs from Worker, worker is not available.";
}
Status GrpcNotifyDistributeWorker::ParseAgentConfigAcquireReply(const proto::AgentConfigAcquireReply &reply,
                                                                DistributedServableConfig *config) {
  MSI_LOG(INFO) << "Success to get Agents configs from Worker, and begin to parser";
  // parser reply message:AgentConfigAcquireReply, parameter:rank_table_content
  config->rank_table_content = reply.rank_table_content();
  // parser reply message:AgentConfigAcquireReply, parameter:rank_list
  for (auto &temp_rank : reply.rank_list()) {
    OneRankConfig ome_rank_config;
    ome_rank_config.ip = temp_rank.ip();
    ome_rank_config.device_id = temp_rank.device_id();
    config->rank_list.push_back(ome_rank_config);
  }
  // parser reply message:AgentConfigAcquireReply, parameter:common_meta
  auto &temp_common_meta = reply.common_meta();
  config->common_meta.servable_name = temp_common_meta.servable_name();
  config->common_meta.with_batch_dim = temp_common_meta.with_batch_dim();
  for (auto &temp_without_batch_dim_inputs : temp_common_meta.without_batch_dim_inputs()) {
    config->common_meta.without_batch_dim_inputs.push_back(temp_without_batch_dim_inputs);
  }
  for (auto &count : temp_common_meta.inputs_count()) {
    config->common_meta.inputs_count[count.first] = count.second;
  }
  for (auto &count : temp_common_meta.outputs_count()) {
    config->common_meta.outputs_count[count.first] = count.second;
  }

  // parser reply message:AgentConfigAcquireReply, parameter:distributed_meta
  auto &temp_distributed_meta = reply.distributed_meta();
  config->distributed_meta.rank_size = temp_distributed_meta.rank_size();
  config->distributed_meta.stage_size = temp_distributed_meta.stage_size();
  MSI_LOG(INFO) << "Success to parser reply message and save to DistributedServableConfig";

  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
