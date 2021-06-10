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

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <future>
#include "common/serving_common.h"
#include "common/servable.h"

namespace mindspore::serving {

using InstanceData = std::vector<TensorBasePtr>;
using TensorsData = std::vector<TensorBasePtr>;
struct Instance;
using InstancePtr = std::shared_ptr<Instance>;

struct WorkerUserContext {
  RequestSpec request_spec;
  MethodSignature method_def;
};

enum InstancePhase {
  kInstancePhaseNone,
  kInstancePhasePipeline,
  kInstancePhasePreprocess,
  kInstancePhasePredict,
  kInstancePhasePostprocess,
  kInstancePhaseDone,
};

struct InstanceContext {
  uint64_t user_id = 0;
  uint32_t instance_index = 0;

  std::promise<void> promise;
  std::shared_ptr<WorkerUserContext> user_context = nullptr;

  bool operator==(const InstanceContext &other) const {
    return user_id == other.user_id && instance_index == other.instance_index;
  }
};

struct Instance {
  InstanceData data;  // for inputs of pipeline, preprocess, predict, postprocess or output

  InstanceData input_data;        // input data
  InstanceData pipeline_data;     // pipeline data
  InstanceData preprocess_data;   // preprocess result
  InstanceData predict_data;      // predict result
  InstanceData postprocess_data;  // postprocess result
  InstanceContext context;
  InstancePhase phase = kInstancePhaseNone;
  Status error_msg = SUCCESS;
};

struct ResultInstance {
  InstanceData data;
  Status error_msg = SUCCESS;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_INSTANCE_H
