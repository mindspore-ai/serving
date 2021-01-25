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

#ifndef MINDSPORE_SERVING_WORKER_AGENT_H
#define MINDSPORE_SERVING_WORKER_AGENT_H
#include <vector>
#include "worker/distributed_worker/agent_executor.h"

namespace mindspore {
namespace serving {
class MS_API WorkerAgent {
 public:
  static WorkerAgent &Instance();
  Status LoadModelFromFile(const AgentStartUpConfig &config);
  Status Clear();

  Status ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply);

 private:
  AgentStartUpConfig config_;
  WorkerAgentExecutor executor_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_AGENT_H
