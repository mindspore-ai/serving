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

#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include "mindspore_serving/ccsrc/worker/model_loader_base.h"
#include "worker/distributed_worker/common.h"
#include "worker/distributed_worker/notify_agent/base_notify_agent.h"

using nlohmann::json;
namespace mindspore {
namespace serving {
struct DistributedAgentContext {
  std::vector<WorkerAgentSpec> agent_spec_;
  std::shared_ptr<BaseNotifyAgent> notify_agent_ = nullptr;
};

class MS_API DistributedModelLoader : public DirectModelLoaderBase {
 public:
  DistributedModelLoader() = default;
  ~DistributedModelLoader();
  // from python, worker.py
  Status LoadModel(const std::string &servable_name, const std::string &rank_table_json_file,
                   uint64_t wait_agents_time_in_seconds);

  // invoke from agent
  Status GetDistributedServableConfig(DistributedServableConfig *config) const;
  // send model and group

  // register and unregister agent, agent_spec_list_
  Status RegisterAgent(const std::vector<WorkerAgentSpec> &agent_specs);
  Status OnAgentExit();

  // predict, use config_ and agent_spec_list_
  Status Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                 uint64_t subgraph = 0) override;

  std::vector<TensorInfo> GetInputInfos(uint64_t subgraph = 0) const override;
  std::vector<TensorInfo> GetOutputInfos(uint64_t subgraph = 0) const override;
  uint64_t GetBatchSize() const override;
  void Clear() override;
  void OnAgentFailed();
  uint64_t GetGraphNum() const override;

  std::string GetModelKey() const { return model_key_; }

 private:
  DistributedServableConfig config_;
  std::atomic_bool config_loaded_ = false;

  std::string model_key_;
  std::atomic_bool model_loaded_ = false;
  uint64_t graph_num_ = 0;
  std::mutex mutex_;
  std::map<uint32_t, DistributedAgentContext> agent_spec_map_;
  std::string rank_table_json_file_;

  std::map<uint64_t, std::vector<TensorInfo>> input_infos_;
  std::map<uint64_t, std::vector<TensorInfo>> output_infos_;
  uint64_t batch_size_;
  bool all_stage_has_input_ = false;

  std::atomic_flag promise_set_flag_ = ATOMIC_FLAG_INIT;
  std::atomic_bool registered_end_flag_ = false;
  std::promise<bool> agents_promise_;

  Status InitConfigOnStartup(const std::string &rank_table_json_file);
  Status WaitAgentsReady(uint64_t wait_agents_time_in_seconds);
  Status CheckAgentsInfosAndInitTensorInfos();
  Status CompareTensorInfos(const std::vector<TensorInfo> &lefts, const std::vector<TensorInfo> &rights);
  Status CheckRankConfig();
  void SetWaitAgentsPromise(bool flag);
  Status PredictInner(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                      uint64_t subgraph = 0);
  // agent stubs
  Status ParserRankTableWithGroupList(const std::string &rank_table_json_file, const json &rank_table_json);

  Status ParserRankTableWithServerList(const std::string &rank_table_json_file, const json &rank_table_json);

  json ParserArrayInJson(const json &json_array, const std::string &str);

  std::string ParserStringInJson(const json &json_str, const std::string &str);
  Status ConvertStr2Int(const std::string &rank_table_json_file, const std::string &para_str,
                        const std::string &para_key, uint32_t *para_int) const;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_DISTRIBUTED_SERVABLE_H
