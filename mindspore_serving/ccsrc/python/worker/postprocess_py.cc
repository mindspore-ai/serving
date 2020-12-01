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

#include "python/worker/postprocess_py.h"
#include <pybind11/embed.h>

namespace mindspore::serving {

size_t PyPostprocess::GetInputsCount(const std::string &postprocess_name) const {
  size_t inputs_count;
  size_t outputs_count;
  (void)PyPostprocessStorage::Instance()->GetPyPostprocessInfo(postprocess_name, &inputs_count, &outputs_count);
  return inputs_count;
}

size_t PyPostprocess::GetOutputsCount(const std::string &postprocess_name) const {
  size_t inputs_count;
  size_t outputs_count;
  (void)PyPostprocessStorage::Instance()->GetPyPostprocessInfo(postprocess_name, &inputs_count, &outputs_count);
  return outputs_count;
}

std::shared_ptr<PyPostprocessStorage> PyPostprocessStorage::Instance() {
  static std::shared_ptr<PyPostprocessStorage> instance;
  if (instance == nullptr) {
    instance = std::make_shared<PyPostprocessStorage>();
  }
  return instance;
}

void PyPostprocessStorage::Register(const std::string &postprocess_name, size_t inputs_count, size_t outputs_count) {
  postprocess_infos_[postprocess_name] = std::make_pair(inputs_count, outputs_count);
  MSI_LOG_INFO << "Register python postprocess " << postprocess_name;
  PostprocessStorage::Instance().Register(postprocess_name, py_postprocess_);
}

bool PyPostprocessStorage::GetPyPostprocessInfo(const std::string &postprocess_name, size_t *inputs_count,
                                                size_t *outputs_count) {
  MSI_EXCEPTION_IF_NULL(inputs_count);
  MSI_EXCEPTION_IF_NULL(outputs_count);
  auto it = postprocess_infos_.find(postprocess_name);
  if (it == postprocess_infos_.end()) {
    return false;
  }
  *inputs_count = it->second.first;
  *outputs_count = it->second.second;
  return true;
}

std::vector<size_t> PyPostprocessStorage::GetPyCppPostprocessInfo(const std::string &postprocess_name) const {
  std::vector<size_t> results;
  auto postprocess = PostprocessStorage::Instance().GetPostprocess(postprocess_name);
  if (!postprocess) {
    return results;
  }
  size_t inputs_count = postprocess->GetInputsCount(postprocess_name);
  size_t outputs_count = postprocess->GetOutputsCount(postprocess_name);
  if (inputs_count == 0 || outputs_count == 0) {
    MSI_LOG_ERROR << "Postprocess " + postprocess_name + " input or output names cannot be null";
    return results;
  }
  results.push_back(inputs_count);
  results.push_back(outputs_count);
  return results;
}

PyPostprocessStorage::PyPostprocessStorage() {}

PyPostprocessStorage::~PyPostprocessStorage() = default;

}  // namespace mindspore::serving
