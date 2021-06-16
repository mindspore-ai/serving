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

#include "python/agent/agent_py.h"
#include "common/exit_handle.h"
#include "worker/distributed_worker/agent_startup.h"
#include "worker/distributed_worker/worker_agent.h"

namespace mindspore::serving {

DistributedServableConfig PyAgent::GetAgentsConfigsFromWorker(const std::string &distributed_address) {
  auto status = WorkerAgentStartUp::Instance().GetAgentsConfigsFromWorker(distributed_address);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }

  DistributedServableConfig config;
  status = WorkerAgentStartUp::Instance().GetDistributedServableConfig(&config);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  return config;
}

void PyAgent::NotifyFailed(const std::string &distributed_address) {
  WorkerAgentStartUp::Instance().NotifyFailed(distributed_address);
}

void PyAgent::StartAgent(const AgentStartUpConfig &start_config, const std::string &dec_key,
                         const std::string &dec_mode) {
  auto status = WorkerAgent::Instance().StartAgent(start_config, dec_key, dec_mode);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

void PyAgent::WaitAndClear() {
  {
    py::gil_scoped_release release;
    ExitSignalHandle::Instance().AgentWait();
  }
  WorkerAgent::Instance().Clear();
  MSI_LOG_INFO << "Python agent end wait and clear";
}

void PyAgent::StopAndClear() {
  ExitSignalHandle::Instance().Stop();
  WorkerAgent::Instance().Clear();
}

void PyAgent::StartupNotifyExit(const std::string &distributed_address, const std::string &agent_ip) {
  WorkerAgentStartUp::Instance().StartupNotifyExit(distributed_address, agent_ip);
}

}  // namespace mindspore::serving
