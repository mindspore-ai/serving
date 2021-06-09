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

#ifndef MINDSPORE_SERVING_WORKER_PIPELINE_H
#define MINDSPORE_SERVING_WORKER_PIPELINE_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "common/serving_common.h"
#include "common/instance.h"

namespace mindspore::serving {

class MS_API PipelineStorage {
 public:
  Status RegisterPipeline(const PipelineSignature &method);
  bool GetPipelineDef(std::vector<PipelineSignature> *pipelines) const;
  bool GetMethodDeclare(const std::string &method_name, PipelineSignature *method) const;

  Status CheckServable(const RequestSpec &request_spec);
  Status Run(const RequestSpec &request_spec, const std::vector<InstanceData> instances_data,
             std::vector<InstancePtr> *out);
  Status RunAsync(const RequestSpec &request_spec, const std::vector<InstanceData> instances_data,
                  std::vector<InstancePtr> *out);
  static PipelineStorage &Instance();

 private:
  std::vector<PipelineSignature> pipelines_;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_PIPELINE_H
