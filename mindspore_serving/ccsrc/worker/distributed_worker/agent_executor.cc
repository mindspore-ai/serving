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
#include "worker/distributed_worker/agent_executor.h"

namespace mindspore {
namespace serving {

Status WorkerAgentExecutor::LoadModelFromFile(const AgentStartUpConfig &config) { return Status(); }
Status WorkerAgentExecutor::UnloadModel() { return Status(); }
Status WorkerAgentExecutor::ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply) {
  return Status();
}
std::vector<serving::TensorInfo> WorkerAgentExecutor::GetInputInfos() const {
  return std::vector<serving::TensorInfo>();
}
std::vector<serving::TensorInfo> WorkerAgentExecutor::GetOutputInfos() const {
  return std::vector<serving::TensorInfo>();
}
ssize_t WorkerAgentExecutor::GetBatchSize() const { return 0; }
}  // namespace serving
}  // namespace mindspore
