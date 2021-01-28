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

#ifndef MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H
#define MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "worker/sevable_base.h"
#include "worker/distributed_worker/common.h"
#include "worker/distributed_worker/notify_agent/base_notify_agent.h"

namespace mindspore {
namespace serving {

struct DistributedAgentContext {
  WorkerAgentSpec agent_spec_;
  std::shared_ptr<BaseNotifyAgent> notify_agent_ = nullptr;
};

class MS_API DistributedServable : public ServableBase {
 public:
  // from python, worker.py
  Status StartServable(const std::string &servable_directory, const std::string &servable_name,
                       const std::string &rank_table_json_file, uint64_t version_number);

  // invoke from agent
  Status GetDistributedServableConfig(DistributedServableConfig *config) const;
  // send model and group

  // register and unregister agent, agent_spec_list_
  Status RegisterAgent(const WorkerAgentSpec &agent_spec);
  Status UnregisterAgent(const WorkerAgentSpec &agent_spec);

  // predict, use config_ and agent_spec_list_
  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) override;

  std::vector<TensorInfo> GetInputInfos() const override;
  std::vector<TensorInfo> GetOutputInfos() const override;
  uint64_t GetBatchSize() const override;
  std::string GetServableName() const override;
  uint64_t GetServableVersion() const override;
  void Clear();

 private:
  DistributedServableConfig config_;
  std::string servable_name_;
  uint64_t version_number_ = 0;
  bool model_loaded_ = false;

  std::map<uint32_t, DistributedAgentContext> agent_spec_map_;
  std::string rank_table_json_file_;

  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfo> output_infos_;
  uint64_t batch_size_ = 0;
  std::promise<void> agents_promise_;

  Status InitConfigOnStartup(const std::string &rank_table_json_file);
  Status WaitAgentsReady();
  Status CheckAgentsInfosAndInitTensorInfos();
  Status CompareTensorInfos(const std::vector<TensorInfo> &lefts, const std::vector<TensorInfo> &rights);
  Status CheckRankConfig();
  // agent stubs
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H
