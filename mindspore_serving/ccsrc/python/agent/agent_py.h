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

#ifndef MINDSPORE_SERVER_AGENT_PY_H
#define MINDSPORE_SERVER_AGENT_PY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>
#include "common/serving_common.h"
#include "worker/distributed_worker/common.h"

namespace py = pybind11;

namespace mindspore {
namespace serving {

class MS_API PyAgent {
 public:
  static void StartAgent(const AgentStartUpConfig &start_config);

  static DistributedServableConfig GetAgentsConfigsFromWorker(const std::string &worker_ip, uint32_t worker_port);
  static void WaitAndClear();
  static void StopAndClear();
  // from start up, not agent
  static void NotifyFailed(const std::string &worker_ip, uint32_t worker_port);
  static void StartupNotifyExit(const std::string &worker_ip, uint32_t worker_port, const std::string &agent_ip);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVER_AGENT_PY_H
