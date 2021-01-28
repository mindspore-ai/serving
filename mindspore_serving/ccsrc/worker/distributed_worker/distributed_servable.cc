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

#include "worker/distributed_worker/distributed_servable.h"
#include <vector>
#include <string>
#include "worker/worker.h"
#include "worker/distributed_worker/notify_agent/notify_agent.h"
namespace mindspore {
namespace serving {

Status DistributedServable::Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) {
  return Status();
}
std::vector<TensorInfo> DistributedServable::GetInputInfos() const { return std::vector<TensorInfo>(); }
std::vector<TensorInfo> DistributedServable::GetOutputInfos() const { return std::vector<TensorInfo>(); }
uint64_t DistributedServable::GetBatchSize() const { return 0; }
Status DistributedServable::GetDistributedServableConfig(DistributedServableConfig *config) { return Status(); }
Status DistributedServable::RegisterAgent(const WorkerAgentSpec &agent_spec) {
  DistributedAgentContext context;
  auto it = agent_spec_list_.find(agent_spec.rank_id);
  if (it != agent_spec_list_.end()) {
    MSI_LOG_WARNING << "rank_id " << agent_spec.rank_id << " has been registered";
    return SUCCESS;
  }
  context.agent_spec_ = agent_spec;
  std::shared_ptr<BaseNotifyAgent> notify_agent = std::make_shared<GrpcNotfiyAgent>(agent_spec.agent_address);
  context.notify_agent_ = notify_agent;
  agent_spec_list_[agent_spec.rank_id] = context;
  if (config_.rank_size == agent_spec_list_.size()) {
    Status status = Worker::GetInstance().RegisterWorker();
    if (status != SUCCESS) {
      Clear();
      return FAILED;
    }
  }
  return SUCCESS;
}

void DistributedServable::Clear() {
  for (auto agent : agent_spec_list_) {
    agent.second.notify_agent_->Exit();
  }
  Worker::GetInstance().StopServable(false);
}

Status DistributedServable::UnregisterAgent(const WorkerAgentSpec &agent_spec) {
  for (auto iter = agent_spec_list_.begin(); iter != agent_spec_list_.end();) {
    if (agent_spec.rank_id == iter->second.agent_spec_.rank_id) {
      iter = agent_spec_list_.erase(iter);
    } else {
      ++iter;
    }
  }
  // todo: send exit message to agent, and then exit if split with master
  Clear();
  return Status();
}

Status DistributedServable::SetProperty(uint32_t rank_size, uint32_t stage_size, bool with_bach_dim,
                                        const std::vector<int> &without_batch_dim_inputs) {
  return Status();
}
Status DistributedServable::InitConfigOnStartup(const std::string &rank_table_json_file) { return Status(); }
}  // namespace serving
}  // namespace mindspore
