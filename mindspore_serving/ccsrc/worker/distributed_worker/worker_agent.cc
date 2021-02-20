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
#include "common/proto_tensor.h"

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
      MSI_LOG_INFO << "End unregister to worker";
    }
    notify_worker_ = nullptr;
  }
  grpc_server_.Stop();
  session_.UnloadModel();
  return SUCCESS;
}

Status WorkerAgent::Run(const proto::DistributedPredictRequest &request, proto::DistributedPredictReply *reply) {
  // todo : DistributedPredictRequest->RequestBase
  // todo : DistributedPredictReply->ReplyBase
  Status status;
  try {
    MSI_TIME_STAMP_START(ExecuteModel)
    // status = session_.ExecuteModel(request_wrap, &reply_wrap);
    MSI_TIME_STAMP_END(ExecuteModel)
  } catch (const std::bad_alloc &ex) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: malloc memory failed";
  } catch (const std::runtime_error &ex) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: runtime error occurred: " << ex.what();
  } catch (const std::exception &ex) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: exception occurred: " << ex.what();
  } catch (...) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: exception occurred";
  }
  if (status != SUCCESS) {
    reply->Clear();
    auto error_msg = reply->mutable_error_msg();
    error_msg->set_error_code(status.StatusCode());
    error_msg->set_error_msg(status.StatusMessage());
  }
  return status;
}

Status WorkerAgent::StartAgent(const AgentStartUpConfig &config) {
  Status status;
  config_ = config;
  const auto &common_meta = config.common_meta;
  status = session_.LoadModelFromFile(kDeviceTypeAscendMS, config.device_id, config.model_file_name, kMindIR,
                                      common_meta.with_batch_dim, common_meta.without_batch_dim_inputs, {});
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "LoadModelFromFile failed, servable name: " << common_meta.servable_name
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
  MSI_LOG_INFO << "Start agent success, servable name: " << common_meta.servable_name << ", rank_id: " << config.rank_id
               << ", device id: " << config.device_id << ", model file: " << config.model_file_name
               << ", rank table file: " << config.rank_table_json_file_name
               << ", group config file: " << config.group_file_name;
  return SUCCESS;
}

Status WorkerAgent::StartGrpcServer() {
  std::string server_address = config_.agent_ip + ":" + std::to_string(config_.agent_port);
  grpc_server_.Start(std::make_shared<MSAgentImpl>(server_address), config_.agent_ip, config_.agent_port,
                     gRpcMaxMBMsgSize, "Agent");
  return SUCCESS;
}

Status WorkerAgent::RegisterAgent() {
  notify_worker_ = std::make_shared<GrpcNotifyDistributeWorker>(config_.worker_ip, config_.worker_port,
                                                                config_.agent_ip, config_.agent_port);
  WorkerAgentSpec spec;
  spec.agent_address = config_.agent_ip + ":" + std::to_string(config_.agent_port);
  spec.rank_id = config_.rank_id;
  spec.batch_size = session_.GetBatchSize();
  spec.input_infos = session_.GetInputInfos();
  spec.output_infos = session_.GetOutputInfos();
  return notify_worker_->Register({spec});
}

void WorkerAgent::StopAgent(bool notify_worker) {
  exit_notify_worker_ = notify_worker;
  ExitSignalHandle::Instance().Stop();
}

}  // namespace serving
}  // namespace mindspore
