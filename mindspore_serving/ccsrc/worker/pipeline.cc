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

#include "worker/pipeline.h"
#include <utility>
#include "worker/worker.h"

namespace mindspore::serving {
Status PipelineStorage::RegisterPipeline(const PipelineSignature &method) {
  MSI_LOG_INFO << "Register Pipeline method " << method.pipeline_name;
  for (auto &item : pipelines_) {
    // cppcheck-suppress useStlAlgorithm
    if (item.pipeline_name == method.pipeline_name) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Pipeline " << method.pipeline_name << " has been registered more than once.";
    }
  }
  pipelines_.push_back(method);
  return SUCCESS;
}
bool PipelineStorage::GetPipelineDef(std::vector<PipelineSignature> *pipelines) const {
  if (pipelines_.size() != 0) {
    *pipelines = pipelines_;
    return true;
  }
  return false;
}
bool PipelineStorage::GetMethodDeclare(const std::string &method_name, PipelineSignature *method) const {
  MSI_EXCEPTION_IF_NULL(method);
  auto item = find_if(pipelines_.begin(), pipelines_.end(),
                      [&](const PipelineSignature &v) { return v.pipeline_name == method_name; });
  if (item != pipelines_.end()) {
    *method = *item;
    return true;
  }
  return false;
}

PipelineStorage &PipelineStorage::Instance() {
  static PipelineStorage storage;
  return storage;
}

Status PipelineStorage::CheckServable(const RequestSpec &request_spec) { return SUCCESS; }

Status PipelineStorage::Run(const RequestSpec &request_spec, const std::vector<InstanceData> instances_data,
                            std::vector<InstancePtr> *out) {
  return Worker::GetInstance().Run(request_spec, instances_data, out);
}

Status PipelineStorage::RunAsync(const RequestSpec &request_spec, const std::vector<InstanceData> instances_data,
                                 std::vector<InstancePtr> *out) {
  return SUCCESS;
}
}  // namespace mindspore::serving
