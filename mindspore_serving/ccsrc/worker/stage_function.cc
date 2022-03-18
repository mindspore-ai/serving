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

#include "worker/stage_function.h"
#include <utility>

namespace mindspore::serving {
bool CppStageFunctionStorage::Register(const std::string &function_name,
                                       std::shared_ptr<CppStageFunctionBase> function) {
  if (function_map_.find(function_name) != function_map_.end()) {
    MSI_LOG_WARNING << "function " << function_name << " has been registered";
    return false;
  }
  function_map_[function_name] = std::move(function);
  return true;
}

void CppStageFunctionStorage::Unregister(const std::string &function_name) {
  auto it = function_map_.find(function_name);
  if (it == function_map_.end()) {
    return;
  }
  (void)function_map_.erase(it);
}

CppStageFunctionStorage &CppStageFunctionStorage::Instance() {
  static CppStageFunctionStorage storage;
  return storage;
}

std::shared_ptr<CppStageFunctionBase> CppStageFunctionStorage::GetFunction(const std::string &func_name) const {
  auto it = function_map_.find(func_name);
  if (it != function_map_.end()) {
    return it->second;
  }
  return nullptr;
}

CppRegStageFunction::CppRegStageFunction(const std::string &function_name,
                                         std::shared_ptr<CppStageFunctionBase> function) {
  func_name_ = function_name;
  MSI_LOG_INFO << "Register C++ function " << function_name;
  register_success_ = CppStageFunctionStorage::Instance().Register(function_name, std::move(function));
}

CppRegStageFunction::~CppRegStageFunction() {
  if (register_success_) {
    MSI_LOG_INFO << "Unregister C++ function " << func_name_;
    CppStageFunctionStorage::Instance().Unregister(func_name_);
  }
}

PyStageFunctionStorage::PyStageFunctionStorage() = default;
PyStageFunctionStorage::~PyStageFunctionStorage() = default;

std::shared_ptr<PyStageFunctionStorage> PyStageFunctionStorage::Instance() {
  static std::shared_ptr<PyStageFunctionStorage> instance;
  if (instance == nullptr) {
    instance = std::make_shared<PyStageFunctionStorage>();
  }
  return instance;
}

void PyStageFunctionStorage::Register(const std::string &func_name, size_t inputs_count, size_t outputs_count) {
  function_infos_[func_name] = std::make_pair(inputs_count, outputs_count);
  MSI_LOG_INFO << "Register python stage function " << func_name << " inputs count " << inputs_count
               << " outputs count " << outputs_count;
}

bool PyStageFunctionStorage::HasPyFunction(const std::string &func_name) {
  auto it = function_infos_.find(func_name);
  return it != function_infos_.end();
}

bool PyStageFunctionStorage::GetPyFunctionInfo(const std::string &func_name, size_t *inputs_count,
                                               size_t *outputs_count) {
  MSI_EXCEPTION_IF_NULL(inputs_count);
  MSI_EXCEPTION_IF_NULL(outputs_count);
  auto it = function_infos_.find(func_name);
  if (it == function_infos_.end()) {
    return false;
  }
  *inputs_count = it->second.first;
  *outputs_count = it->second.second;
  return true;
}

std::vector<size_t> PyStageFunctionStorage::GetPyCppFunctionInfo(const std::string &func_name) const {
  size_t inputs_count = 0;
  size_t outputs_count = 0;
  if (PyStageFunctionStorage::Instance()->GetPyFunctionInfo(func_name, &inputs_count, &outputs_count)) {
    return {inputs_count, outputs_count};
  }
  auto function = CppStageFunctionStorage::Instance().GetFunction(func_name);
  if (!function) {
    return {};
  }
  inputs_count = function->GetInputsCount(func_name);
  outputs_count = function->GetOutputsCount(func_name);
  if (inputs_count == 0 || outputs_count == 0) {
    MSI_LOG_ERROR << "Call " + func_name + " inputs or outputs count cannot be 0";
    return {};
  }
  return {inputs_count, outputs_count};
}
}  // namespace mindspore::serving
