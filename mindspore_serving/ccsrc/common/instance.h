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

#ifndef MINDSPORE_SERVING_INSTANCE_H
#define MINDSPORE_SERVING_INSTANCE_H

#include <map>
#include <memory>
#include "common/serving_common.h"
#include "common/servable.h"
#include "common/instance_data.h"

namespace mindspore::serving {

struct Instance {
  InstanceData data;  // for inputs of function, predict, output

  const MethodSignature *method_def = nullptr;
  uint64_t stage_index = 0;
  uint64_t stage_max = 0;
  std::map<size_t, InstanceData> stage_data_list;  // input: 0, stage: 1-n

  uint64_t user_id = 0;
  Status error_msg = SUCCESS;
};

using InstancePtr = std::shared_ptr<Instance>;

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_INSTANCE_H
