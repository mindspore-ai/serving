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
#include "worker/distributed_worker/agent_startup.h"
#include <fstream>
#include "worker/distributed_worker/notify_distributed/notify_worker.h"
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {

WorkerAgentStartUp &WorkerAgentStartUp::Instance() {
  static WorkerAgentStartUp instance;
  return instance;
}

Status WorkerAgentStartUp::GetAgentsConfigsFromWorker(const std::string &worker_ip, uint32_t worker_port) {
  return GrpcNotifyDistributeWorker::GetAgentsConfigsFromWorker(worker_ip, worker_port, &config_);
}

Status WorkerAgentStartUp::GetDistributedServableConfig(DistributedServableConfig *config) {
  MSI_EXCEPTION_IF_NULL(config);
  if (config_.rank_list.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Rank table config is not ready";
  }
  *config = config_;
  return SUCCESS;
}

Status WorkerAgentStartUp::NotifyFailed(const std::string &worker_ip, uint32_t worker_port) {
  return GrpcNotifyDistributeWorker::NotifyFailed(worker_ip, worker_port);
}

void WorkerAgentStartUp::StartupNotifyExit(const std::string &worker_ip, uint32_t worker_port,
                                           const std::string &agent_ip) {
  GrpcNotifyDistributeWorker::StartupNotifyExit(worker_ip, worker_port, agent_ip);
}

}  // namespace serving
}  // namespace mindspore
