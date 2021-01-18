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

#ifndef MINDSPORE_SERVING_WORKER_AGENT_EXECUTOR_H
#define MINDSPORE_SERVING_WORKER_AGENT_EXECUTOR_H

#include <vector>
#include "common/serving_common.h"
#include "worker/inference/inference.h"
#include "worker/distributed_worker/common.h"

namespace mindspore {
namespace serving {
class MS_API WorkerAgentExecutor {
 public:
  // from python
  Status LoadModelFromFile(const AgentStartUpConfig &config);
  // ctrl+c, worker exit
  Status UnloadModel();

  // from worker
  Status ExecuteModel(const std::vector<TensorBasePtr> &request, std::vector<TensorBasePtr> *reply);

  // for register
  std::vector<serving::TensorInfo> GetInputInfos() const;

  std::vector<serving::TensorInfo> GetOutputInfos() const;

  ssize_t GetBatchSize() const;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_AGENT_EXECUTOR_H
