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

#ifndef MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H
#define MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H

#include <vector>
#include <string>
#include <map>
#include "worker/model.h"
#include "worker/distributed_worker/common.h"

namespace mindspore {
namespace serving {

class MS_API DistributedServable : public ServableBase {
 public:
  // from python, servable_config.py
  Status SetProperty(uint32_t rank_size, uint32_t stage_size, bool with_bach_dim,
                     const std::vector<int> &without_batch_dim_inputs);
  // from python, worker.py
  Status InitConfigOnStartup(const std::string &rank_table_json_file);
  // invoke from agent
  Status GetDistributedServableConfig(DistributedServableConfig *config);
  // send model and group

  // register and unregister agent, agent_spec_list_
  Status RegisterAgent(const WorkerAgentSpec &agent_spec);
  Status UnregisterAgent(const WorkerAgentSpec &agent_spec);

  // predict, use config_ and agent_spec_list_
  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) override;

  std::vector<TensorInfo> GetInputInfos() const override;
  std::vector<TensorInfo> GetOutputInfos() const override;
  uint64_t GetBatchSize() const override;

 private:
  DistributedServableConfig config_;
  std::map<uint32_t, WorkerAgentSpec> agent_spec_list_;
  // agent stubs
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H
