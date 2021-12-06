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

#include "common/servable.h"
#include <set>
#include <sstream>
#include "worker/stage_function.h"

namespace mindspore::serving {

void LocalModelMeta::SetModelFormat(const std::string &format) {
  if (format == "om") {
    model_format = kOM;
  } else if (format == "mindir") {
    model_format = kMindIR;
  } else if (format == "mindir_opt") {
    model_format = kMindIR_Opt;
  } else {
    MSI_LOG_ERROR << "Invalid model format " << format;
  }
}

std::string ServableLoadSpec::Repr() const {
  std::string version;
  if (version_number > 0) {
    version = " version(" + std::to_string(version_number) + ") ";
  }
  return "servable(" + servable_name + ") " + version;
}

std::string WorkerRegSpec::Repr() const {
  std::stringstream str_stream;
  str_stream << "{worker_pid:" << worker_pid << ", address:" + worker_address
             << ", servable:" << servable_spec.servable_name + ", version:" << servable_spec.version_number << "}";
  return str_stream.str();
}

std::string RequestSpec::Repr() const {
  std::string version;
  if (version_number > 0) {
    version = " version(" + std::to_string(version_number) + ") ";
  }
  return "servable(" + servable_name + ") " + "method(" + method_name + ") " + version;
}

void MethodSignature::AddStageFunction(const std::string &func_name,
                                       const std::vector<std::pair<size_t, uint64_t>> &stage_inputs,
                                       uint64_t batch_size, const std::string &tag) {
  MethodStage stage;
  stage.method_name = method_name;
  stage.stage_index = stage_index;
  stage.stage_key = func_name;

  if (PyStageFunctionStorage::Instance()->HasPyFunction(func_name)) {
    stage.stage_type = kMethodStageTypePyFunction;
  } else {
    auto func = CppStageFunctionStorage::Instance().GetFunction(func_name);
    if (!func) {
      MSI_LOG_EXCEPTION << "Function '" << func_name << "' is not defined";
    }
    stage.stage_type = kMethodStageTypeCppFunction;
  }
  stage.stage_inputs = stage_inputs;
  stage.batch_size = batch_size;
  if (tag.empty()) {
    stage.tag = "Function '" + func_name + "'";
  } else {
    stage.tag = tag;
  }
  stage_map[stage_index] = stage;
  stage_index += 1;
}

void MethodSignature::AddStageModel(const std::string &model_key,
                                    const std::vector<std::pair<size_t, uint64_t>> &stage_inputs, uint64_t subgraph,
                                    const std::string &tag) {
  MethodStage stage;
  stage.method_name = method_name;
  stage.stage_index = stage_index;
  stage.stage_key = model_key;
  stage.stage_type = kMethodStageTypeModel;
  stage.stage_inputs = stage_inputs;
  stage.subgraph = subgraph;
  if (tag.empty()) {
    stage.tag = "Model '" + model_key + "'";
  } else {
    stage.tag = tag;
  }
  stage_map[stage_index] = stage;
  stage_index += 1;
}

void MethodSignature::SetReturn(const std::vector<std::pair<size_t, uint64_t>> &return_inputs) {
  MethodStage stage;
  stage.method_name = method_name;
  stage.stage_index = stage_index;
  stage.stage_key = "return";
  stage.stage_type = kMethodStageTypeReturn;
  stage.stage_inputs = return_inputs;
  stage_map[stage_index] = stage;
}

size_t MethodSignature::GetStageMax() const { return stage_index; }

const MethodSignature *ServableSignature::GetMethodDeclare(const std::string &method_name) const {
  auto item =
    find_if(methods.begin(), methods.end(), [&](const MethodSignature &v) { return v.method_name == method_name; });
  if (item == methods.end()) {
    return nullptr;
  }
  return &(*item);
}

const ModelMeta *ServableSignature::GetModelDeclare(const std::string &model_key) const {
  auto item = find_if(model_metas.begin(), model_metas.end(),
                      [&](const ModelMeta &v) { return v.common_meta.model_key == model_key; });
  if (item == model_metas.end()) {
    return nullptr;
  }
  return &(*item);
}

}  // namespace mindspore::serving
