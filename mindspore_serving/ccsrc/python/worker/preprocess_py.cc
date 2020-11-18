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

#include "python/worker/preprocess_py.h"
#include <pybind11/embed.h>
#include <memory>
#include <vector>
#include <string>
#include <utility>

namespace mindspore::serving {

size_t PyPreprocess::GetInputsCount(const std::string &preprocess_name) const {
  size_t inputs_count;
  size_t outputs_count;
  (void)PyPreprocessStorage::Instance()->GetPyPreprocessInfo(preprocess_name, inputs_count, outputs_count);
  return inputs_count;
}

size_t PyPreprocess::GetOutputsCount(const std::string &preprocess_name) const {
  size_t inputs_count;
  size_t outputs_count;
  (void)PyPreprocessStorage::Instance()->GetPyPreprocessInfo(preprocess_name, inputs_count, outputs_count);
  return outputs_count;
}

std::shared_ptr<PyPreprocessStorage> PyPreprocessStorage::Instance() {
  static std::shared_ptr<PyPreprocessStorage> instance;
  if (instance == nullptr) {
    instance = std::make_shared<PyPreprocessStorage>();
  }
  return instance;
}

void PyPreprocessStorage::Register(const std::string &preprocess_name, size_t inputs_count, size_t outputs_count) {
  preprocess_infos_[preprocess_name] = std::make_pair(inputs_count, outputs_count);
  MSI_LOG_INFO << "Register python preprocess " << preprocess_name;
  PreprocessStorage::Instance().Register(preprocess_name, py_preprocess_);
}

bool PyPreprocessStorage::GetPyPreprocessInfo(const std::string &preprocess_name, size_t &inputs_count,
                                              size_t &outputs_count) {
  auto it = preprocess_infos_.find(preprocess_name);
  if (it == preprocess_infos_.end()) {
    return false;
  }
  inputs_count = it->second.first;
  outputs_count = it->second.second;
  return true;
}

std::vector<size_t> PyPreprocessStorage::GetPyCppPreprocessInfo(const std::string &preprocess_name) const {
  std::vector<size_t> results;
  auto preprocess = PreprocessStorage::Instance().GetPreprocess(preprocess_name);
  if (!preprocess) {
    return results;
  }
  size_t inputs_count = preprocess->GetInputsCount(preprocess_name);
  size_t outputs_count = preprocess->GetOutputsCount(preprocess_name);
  if (inputs_count == 0 || outputs_count == 0) {
    MSI_LOG_ERROR << "Preprocess " + preprocess_name + " inputs or outputs count cannot be 0";
    return results;
  }
  results.push_back(inputs_count);
  results.push_back(outputs_count);
  return results;
}

PyPreprocessStorage::PyPreprocessStorage() {}

PyPreprocessStorage::~PyPreprocessStorage() = default;

}  // namespace mindspore::serving
