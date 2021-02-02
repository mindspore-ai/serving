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
#include "worker/distributed_worker/worker_agent.h"
#include <memory>
#include "worker/distributed_worker/agent_process/agent_process.h"
#include "worker/distributed_worker/notify_distributed/notify_worker.h"
#include "common/exit_handle.h"

namespace mindspore {
namespace serving {

WorkerAgent &WorkerAgent::Instance() {
  static WorkerAgent instance;
  return instance;
}

Status WorkerAgent::Clear() {
  if (notify_worker_) {
    if (exit_notify_worker_) {
      notify_worker_->Unregister();
    }
    notify_worker_ = nullptr;
  }
  grpc_server_.Stop();
  executor_.UnloadModel();
  return SUCCESS;
}

Status WorkerAgent::Run(const proto::DistributedPredictRequest &request, proto::DistributedPredictReply *reply) {
  // todo : DistributedPredictRequest->RequestBase
  // todo : DistributedPredictReply->ReplyBase
  return SUCCESS;
}

Status WorkerAgent::StartAgent(const AgentStartUpConfig &config) {
  Status status;
  config_ = config;
  status = executor_.LoadModelFromFile(config);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "LoadModelFromFile failed, servable name: " << config.common_meta.servable_name
                  << ", rank_id: " << config.rank_id << ", device id: " << config.device_id
                  << ", model file: " << config.model_file_name
                  << ", rank table file: " << config.rank_table_json_file_name
                  << ", group config file: " << config.group_file_name;
    return status;
  }
  status = StartGrpcServer();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Start agent grpc server failed, agent ip: " << config.agent_ip
                  << ", agent port: " << config.agent_port;
    return status;
  }
  status = RegisterAgent();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register agent failed, agent ip: " << config.agent_ip << ", agent port: " << config.agent_port
                  << ", worker ip: " << config.worker_ip << ", worker port: " << config.worker_port;
    return status;
  }
  MSI_LOG_INFO << "Start agent success, servable name: " << config.common_meta.servable_name
               << ", rank_id: " << config.rank_id << ", device id: " << config.device_id
               << ", model file: " << config.model_file_name
               << ", rank table file: " << config.rank_table_json_file_name
               << ", group config file: " << config.group_file_name;
  return SUCCESS;
}

Status WorkerAgent::StartGrpcServer() {
  grpc_server_.Start(std::make_shared<MSAgentImpl>(), config_.agent_ip, config_.agent_port, gRpcMaxMBMsgSize, "Agent");
  return SUCCESS;
}

Status WorkerAgent::RegisterAgent() {
  notify_worker_ = std::make_shared<GrpcNotifyDistributeWorker>(config_.worker_ip, config_.agent_port, config_.agent_ip,
                                                                config_.agent_port);
  WorkerAgentSpec spec;
  spec.agent_address = config_.agent_ip + ":" + std::to_string(config_.agent_port);
  spec.rank_id = config_.rank_id;
  spec.batch_size = executor_.GetBatchSize();
  spec.input_infos = executor_.GetInputInfos();
  spec.output_infos = executor_.GetOutputInfos();
  return notify_worker_->Register({spec});
}

void WorkerAgent::StopAgent(bool notify_worker) {
  exit_notify_worker_ = notify_worker;
  ExitSignalHandle::Instance().Stop();
}

}  // namespace serving
}  // namespace mindspore
