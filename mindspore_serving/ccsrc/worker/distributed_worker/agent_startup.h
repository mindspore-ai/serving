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

#ifndef MINDSPORE_SERVING_WORKER_AGENT_STARTUP_H
#define MINDSPORE_SERVING_WORKER_AGENT_STARTUP_H
#include <vector>
#include <string>
#include "common/serving_common.h"
#include "worker/distributed_worker/common.h"
#include "worker/inference/inference.h"

namespace mindspore {
namespace serving {

class MS_API WorkerAgentStartUp {
 public:
  // from python, worker_agent.py
  // start_worker_agent
  // step1, get agents config from worker
  Status InitAgentsConfig(const std::string &model_dir, const std::string &model_file_prefix,
                          const std::string &group_file_dir, const std::string &group_file_prefix);

  Status GetAgentsConfigsFromWorker(const std::string &agent_ip, uint32_t agent_start_port,
                                    const std::string &worker_ip, uint32_t worker_port);
  // step2, invoke from python, get current machine agents config
  Status GetCurrentMachineConfigs(std::vector<AgentStartUpConfig> *configs);

 private:
  DistributedServableConfig config_;
  std::string worker_address_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_AGENT_STARTUP_H
