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
#include <map>
#include "worker/preprocess.h"
#include "worker/postprocess.h"

namespace mindspore::serving {

std::string ServableMeta::Repr() const {
  std::ostringstream stream;
  switch (servable_type) {
    case kServableTypeUnknown:
      stream << "undeclared servable, servable name: '" << common_meta.servable_name << "'";
      break;
    case kServableTypeLocal:
      stream << "local servable, servable name: '" << common_meta.servable_name << "', file: '"
             << local_meta.servable_file + "'";
      break;
    case kServableTypeDistributed:
      stream << "distributed servable, servable name: '" << common_meta.servable_name
             << "', rank size: " << distributed_meta.rank_size << ", stage size " << distributed_meta.stage_size;
      break;
  }
  return stream.str();
}

void LocalServableMeta::SetModelFormat(const std::string &format) {
  if (format == "om") {
    model_format = kOM;
  } else if (format == "mindir") {
    model_format = kMindIR;
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

Status ServableSignature::CheckPreprocessInput(const MethodSignature &method, size_t *preprocess_outputs_count) const {
  std::string model_str = servable_meta.Repr();
  const auto &preprocess_name = method.preprocess_name;
  if (!preprocess_name.empty()) {
    auto preprocess = PreprocessStorage::Instance().GetPreprocess(preprocess_name);
    if (preprocess == nullptr) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Model " << model_str << " method " << method.method_name
                                            << " preprocess " << preprocess_name << " not defined";
    }
    *preprocess_outputs_count = preprocess->GetOutputsCount(preprocess_name);

    for (size_t i = 0; i < method.preprocess_inputs.size(); i++) {
      auto &input = method.preprocess_inputs[i];
      if (input.first != kPredictPhaseTag_Input) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the data of preprocess " << i
               << "th input cannot not come from '" << input.first << "'";
      }
      if (input.second >= method.inputs.size()) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the preprocess " << i
               << "th input uses method " << input.second << "th input, that is greater than the method inputs size "
               << method.inputs.size();
      }
    }
  }
  return SUCCESS;
}

Status ServableSignature::CheckPredictInput(const MethodSignature &method, size_t preprocess_outputs_count) const {
  std::string model_str = servable_meta.Repr();

  for (size_t i = 0; i < method.servable_inputs.size(); i++) {
    auto &input = method.servable_inputs[i];
    if (input.first == kPredictPhaseTag_Input) {
      if (input.second >= method.inputs.size()) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the servable " << i
               << "th input uses method " << input.second << "th input, that is greater than the method inputs size "
               << method.inputs.size();
      }
    } else if (input.first == kPredictPhaseTag_Preproces) {
      if (input.second >= preprocess_outputs_count) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the servable " << i
               << "th input uses preprocess " << input.second
               << "th output, that is greater than the preprocess outputs size " << preprocess_outputs_count;
      }
    } else {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Model " << model_str << " method " << method.method_name << ", the data of servable " << i
             << "th input cannot not come from '" << input.first << "'";
    }
  }
  return SUCCESS;
}

Status ServableSignature::CheckPostprocessInput(const MethodSignature &method, size_t preprocess_outputs_count,
                                                size_t *postprocess_outputs_count) const {
  std::string model_str = servable_meta.Repr();
  const auto &common_meta = servable_meta.common_meta;

  const auto &postprocess_name = method.postprocess_name;
  if (!method.postprocess_name.empty()) {
    auto postprocess = PostprocessStorage::Instance().GetPostprocess(postprocess_name);
    if (postprocess == nullptr) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Model " << model_str << " method " << method.method_name
                                            << " postprocess " << postprocess_name << " not defined";
    }
    *postprocess_outputs_count = postprocess->GetOutputsCount(postprocess_name);

    for (size_t i = 0; i < method.postprocess_inputs.size(); i++) {
      auto &input = method.postprocess_inputs[i];
      if (input.first == kPredictPhaseTag_Input) {
        if (input.second >= method.inputs.size()) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                 << "th input uses method " << input.second << "th input, that is greater than the method inputs size "
                 << method.inputs.size();
        }
      } else if (input.first == kPredictPhaseTag_Preproces) {
        if (input.second >= preprocess_outputs_count) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                 << "th input uses preprocess " << input.second
                 << "th output, that is greater than the preprocess outputs size " << preprocess_outputs_count;
        }
      } else if (input.first == kPredictPhaseTag_Predict) {
        if (input.second >= common_meta.outputs_count) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                 << "th input uses servable " << input.second
                 << "th output, that is greater than the servable outputs size " << common_meta.outputs_count;
        }
      } else {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the data of postprocess " << i
               << "th input cannot not come from '" << input.first << "'";
      }
    }
  }
  return SUCCESS;
}

Status ServableSignature::CheckReturn(const MethodSignature &method, size_t preprocess_outputs_count,
                                      size_t postprocess_outputs_count) const {
  std::string model_str = servable_meta.Repr();
  const auto &common_meta = servable_meta.common_meta;

  for (size_t i = 0; i < method.returns.size(); i++) {
    auto &input = method.returns[i];
    if (input.first == kPredictPhaseTag_Input) {
      if (input.second >= method.inputs.size()) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the method " << i
               << "th output uses method " << input.second << "th input, that is greater than the method inputs size "
               << method.inputs.size();
      }
    } else if (input.first == kPredictPhaseTag_Preproces) {
      if (input.second >= preprocess_outputs_count) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the method " << i
               << "th output uses preprocess " << input.second
               << "th output, that is greater than the preprocess outputs size " << preprocess_outputs_count;
      }
    } else if (input.first == kPredictPhaseTag_Predict) {
      if (input.second >= common_meta.outputs_count) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the method " << i
               << "th output uses servable " << input.second
               << "th output, that is greater than the servable outputs size " << common_meta.outputs_count;
      }
    } else if (input.first == kPredictPhaseTag_Postprocess) {
      if (input.second >= postprocess_outputs_count) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Model " << model_str << " method " << method.method_name << ", the method " << i
               << "th output uses postprocess " << input.second
               << "th output, that is greater than the postprocess outputs size " << postprocess_outputs_count;
      }
    } else {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Model " << model_str << " method " << method.method_name << ", the data of method " << i
             << "th output cannot not come from '" << input.first << "'";
    }
  }
  return SUCCESS;
}

Status ServableSignature::Check() const {
  std::set<std::string> method_set;
  Status status;
  for (auto &method : methods) {
    if (method_set.count(method.method_name) > 0) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Model " << servable_meta.Repr() << " " << method.method_name << " has been defined repeatedly";
    }
    method_set.emplace(method.method_name);

    size_t preprocess_outputs_count = 0;
    size_t postprocess_outputs_count = 0;
    status = CheckPreprocessInput(method, &preprocess_outputs_count);
    if (status != SUCCESS) {
      return status;
    }
    status = CheckPredictInput(method, preprocess_outputs_count);
    if (status != SUCCESS) {
      return status;
    }
    status = CheckPostprocessInput(method, preprocess_outputs_count, &postprocess_outputs_count);
    if (status != SUCCESS) {
      return status;
    }
    status = CheckReturn(method, preprocess_outputs_count, postprocess_outputs_count);
    if (status != SUCCESS) {
      return status;
    }
  }
  return SUCCESS;
}

bool ServableSignature::GetMethodDeclare(const std::string &method_name, MethodSignature *method) {
  MSI_EXCEPTION_IF_NULL(method);
  auto item =
    find_if(methods.begin(), methods.end(), [&](const MethodSignature &v) { return v.method_name == method_name; });
  if (item != methods.end()) {
    *method = *item;
    return true;
  }
  return false;
}

void ServableStorage::Register(const ServableSignature &def) {
  auto model_name = def.servable_meta.common_meta.servable_name;
  if (servable_signatures_map_.find(model_name) == servable_signatures_map_.end()) {
    MSI_LOG_WARNING << "Servable " << model_name << " has already been defined";
  }
  servable_signatures_map_[model_name] = def;
}

bool ServableStorage::GetServableDef(const std::string &model_name, ServableSignature *def) const {
  MSI_EXCEPTION_IF_NULL(def);
  auto it = servable_signatures_map_.find(model_name);
  if (it == servable_signatures_map_.end()) {
    return false;
  }
  *def = it->second;
  return true;
}

ServableStorage &ServableStorage::Instance() {
  static ServableStorage storage;
  return storage;
}

Status ServableStorage::RegisterMethod(const MethodSignature &method) {
  MSI_LOG_INFO << "Declare method " << method.method_name << ", servable " << method.servable_name;
  auto it = servable_signatures_map_.find(method.servable_name);
  if (it == servable_signatures_map_.end()) {
    ServableSignature signature;
    signature.methods.push_back(method);
    servable_signatures_map_[method.servable_name] = signature;
    return SUCCESS;
  }
  for (auto &item : it->second.methods) {
    // cppcheck-suppress useStlAlgorithm
    if (item.method_name == method.method_name) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Method " << method.method_name << " has been registered more than once.";
    }
  }
  it->second.methods.push_back(method);
  return SUCCESS;
}

Status ServableStorage::DeclareServable(ServableMeta servable) {
  auto &common_meta = servable.common_meta;
  MSI_LOG_INFO << "Declare servable " << common_meta.servable_name;
  servable.servable_type = kServableTypeLocal;
  if (servable.local_meta.servable_file.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare servable " << common_meta.servable_name << " failed, servable_file cannot be empty";
  }
  if (servable.local_meta.model_format == ModelType::kUnknownType) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare servable " << common_meta.servable_name << " failed, model_format is not inited";
  }
  auto it = servable_signatures_map_.find(common_meta.servable_name);
  if (it == servable_signatures_map_.end()) {
    ServableSignature signature;
    signature.servable_meta = servable;
    servable_signatures_map_[common_meta.servable_name] = signature;
    return SUCCESS;
  }
  auto &org_servable_meta = it->second.servable_meta;
  if (org_servable_meta.servable_type != kServableTypeUnknown) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Servable " << common_meta.servable_name << " has already been declared as: " << servable.Repr();
  }
  org_servable_meta = servable;
  return SUCCESS;
}

Status ServableStorage::DeclareDistributedServable(ServableMeta servable) {
  auto &common_meta = servable.common_meta;
  MSI_LOG_INFO << "Declare servable " << common_meta.servable_name;
  servable.servable_type = kServableTypeDistributed;
  if (servable.distributed_meta.rank_size == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare distributed servable " << common_meta.servable_name << " failed, rank_size cannot be 0";
  }
  if (servable.distributed_meta.stage_size == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare distributed servable " << common_meta.servable_name << " failed, stage_size cannot be 0";
  }
  auto it = servable_signatures_map_.find(common_meta.servable_name);
  if (it == servable_signatures_map_.end()) {
    ServableSignature signature;
    signature.servable_meta = servable;
    servable_signatures_map_[common_meta.servable_name] = signature;
    return SUCCESS;
  }
  auto &org_servable_meta = it->second.servable_meta;
  if (org_servable_meta.servable_type != kServableTypeUnknown) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Servable " << common_meta.servable_name << " has already been declared as: " << servable.Repr();
  }
  org_servable_meta = servable;
  return SUCCESS;
}

Status ServableStorage::RegisterInputOutputInfo(const std::string &servable_name, size_t inputs_count,
                                                size_t outputs_count) {
  auto it = servable_signatures_map_.find(servable_name);
  if (it == servable_signatures_map_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "RegisterInputOutputInfo failed, cannot find servable " << servable_name;
  }
  auto &servable_meta = it->second.servable_meta;
  auto &common_meta = servable_meta.common_meta;
  if (common_meta.inputs_count != 0 && common_meta.inputs_count != inputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "RegisterInputOutputInfo failed, inputs count " << inputs_count << " not match old count "
           << common_meta.inputs_count << ",servable name " << servable_name;
  }
  if (common_meta.outputs_count != 0 && common_meta.outputs_count != outputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "RegisterInputOutputInfo failed, outputs count " << outputs_count << " not match old count "
           << common_meta.outputs_count << ",servable name " << servable_name;
  }
  common_meta.inputs_count = inputs_count;
  common_meta.outputs_count = outputs_count;
  return SUCCESS;
}

std::vector<size_t> ServableStorage::GetInputOutputInfo(const std::string &servable_name) const {
  std::vector<size_t> result;
  auto it = servable_signatures_map_.find(servable_name);
  if (it == servable_signatures_map_.end()) {
    return result;
  }
  result.push_back(it->second.servable_meta.common_meta.inputs_count);
  result.push_back(it->second.servable_meta.common_meta.outputs_count);
  return result;
}

void ServableStorage::Clear() { servable_signatures_map_.clear(); }

}  // namespace mindspore::serving
