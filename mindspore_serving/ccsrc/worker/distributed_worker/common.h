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

#ifndef MINDSPORE_SERVING_DISTRIBUTED_WORKER_COMMON_H
#define MINDSPORE_SERVING_DISTRIBUTED_WORKER_COMMON_H

#include <vector>
#include <string>
#include <map>
#include "common/serving_common.h"
#include "worker/inference/inference.h"

namespace mindspore {
namespace serving {

struct OneRankConfig {
  std::string ip;
  uint32_t device_id = 0;
};

struct DistributedServableCommonConfig {
  bool with_batch_dim;
  std::vector<int> without_batch_dim_inputs;
};

struct DistributedServableConfig {
  uint32_t rank_size = 0;
  uint32_t stage_size = 0;
  const std::string models_dir;
  const std::string groups_dir;
  std::string rank_table_content;
  std::vector<OneRankConfig> rank_list;
  DistributedServableCommonConfig common_config;
};

struct AgentStartUpConfig {
  uint32_t rank_id;
  uint32_t device_id;
  std::string model_file_name;
  std::string group_file_name;
  std::string rank_table_json_file_name;

  std::string agent_ip;
  uint32_t agent_port;
  std::string worker_ip;
  uint32_t worker_port;

  DistributedServableCommonConfig common_config;
  std::map<std::string, std::string> other_options;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_DISTRIBUTED_WORKER_COMMON_H
