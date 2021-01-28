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
#include "common/servable.h"

namespace mindspore {
namespace serving {

struct OneRankConfig {
  std::string ip;
  uint32_t device_id = 0;
};

struct DistributedServableConfig {
  std::string rank_table_content;
  std::vector<OneRankConfig> rank_list;

  CommonServableMeta common_meta;
  DistributedServableMeta distributed_meta;
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

  CommonServableMeta common_meta;
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_DISTRIBUTED_WORKER_COMMON_H
