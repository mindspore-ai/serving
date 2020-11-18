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
#include "worker/preprocess.h"
#include "worker/postprocess.h"

namespace mindspore::serving {

std::string ServableMeta::Repr() const {
  std::ostringstream stream;
  stream << "path(" << servable_name << ") file(" << servable_file + ")";
  return stream.str();
}

void ServableMeta::SetModelFormat(const std::string &format) {
  if (format == "om") {
    model_format = kOM;
  } else if (format == "mindir") {
    model_format = kMindIR;
  } else {
    MSI_LOG_ERROR << "Invalid model format " << format;
  }
}

std::string LoadServableSpec::Repr() const {
  std::string version;
  if (version_number > 0) {
    version = " version(" + std::to_string(version_number) + ") ";
  }
  return "servable(" + servable_name + ") " + version;
}

std::string WorkerSpec::Repr() const {
  std::string version;
  if (version_number > 0) {
    version = " version(" + std::to_string(version_number) + ") ";
  }
  return "servable(" + servable_name + ") " + version + " address(" + worker_address + ") ";
}

std::string RequestSpec::Repr() const {
  std::string version;
  if (version_number > 0) {
    version = " version(" + std::to_string(version_number) + ") ";
  }
  return "servable(" + servable_name + ") " + "method(" + method_name + ") " + version;
}

Status ServableSignature::Check() const {
  std::set<std::string> method_set;
  std::string model_str = servable_meta.Repr();

  for (auto &method : methods) {
    if (method_set.count(method.method_name) > 0) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Model " << model_str << " " << method.method_name << " has been defined repeatly";
    }
    method_set.emplace(method.method_name);

    size_t preprocess_outputs_count = 0;
    size_t postprocess_outputs_count = 0;

    const auto &preprocess_name = method.preprocess_name;
    if (!preprocess_name.empty()) {
      auto preprocess = PreprocessStorage::Instance().GetPreprocess(preprocess_name);
      if (preprocess == nullptr) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Model " << model_str << " method " << method.method_name
                                              << " preprocess " << preprocess_name << " not defined";
      }
      preprocess_outputs_count = preprocess->GetOutputsCount(preprocess_name);

      for (size_t i = 0; i < method.preprocess_inputs.size(); i++) {
        auto &input = method.preprocess_inputs[i];
        if (input.first != kPredictPhaseTag_Input) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the preprocess " << i
                 << "th input phase tag " << input.first << " invalid";
        }
        if (input.second >= method.inputs.size()) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the preprocess " << i
                 << "th input uses method " << input.second << "th input, that is greater than the method inputs size "
                 << method.inputs.size();
        }
      }
    }

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
               << "Model " << model_str << " method " << method.method_name << ", the servable " << i
               << "th input phase tag " << input.first << " invalid";
      }
    }

    const auto &postprocess_name = method.postprocess_name;
    if (!method.postprocess_name.empty()) {
      auto postprocess = PostprocessStorage::Instance().GetPostprocess(postprocess_name);
      if (postprocess == nullptr) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Model " << model_str << " method " << method.method_name
                                              << " postprocess " << postprocess_name << " not defined";
      }
      postprocess_outputs_count = postprocess->GetOutputsCount(postprocess_name);

      for (size_t i = 0; i < method.postprocess_inputs.size(); i++) {
        auto &input = method.postprocess_inputs[i];
        if (input.first == kPredictPhaseTag_Input) {
          if (input.second >= method.inputs.size()) {
            return INFER_STATUS_LOG_ERROR(FAILED)
                   << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                   << "th input uses method " << input.second
                   << "th input, that is greater than the method inputs size " << method.inputs.size();
          }
        } else if (input.first == kPredictPhaseTag_Preproces) {
          if (input.second >= preprocess_outputs_count) {
            return INFER_STATUS_LOG_ERROR(FAILED)
                   << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                   << "th input uses preprocess " << input.second
                   << "th output, that is greater than the preprocess outputs size " << preprocess_outputs_count;
          }
        } else if (input.first == kPredictPhaseTag_Predict) {
          if (input.second >= servable_meta.outputs_count) {
            return INFER_STATUS_LOG_ERROR(FAILED)
                   << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                   << "th input uses servable " << input.second
                   << "th output, that is greater than the servable outputs size " << servable_meta.outputs_count;
          }
        } else {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the postprocess " << i
                 << "th input phase tag " << input.first << " invalid";
        }
      }
    }
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
        if (input.second >= servable_meta.outputs_count) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Model " << model_str << " method " << method.method_name << ", the method " << i
                 << "th output uses servable " << input.second
                 << "th output, that is greater than the servable outputs size " << servable_meta.outputs_count;
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
               << "Model " << model_str << " method " << method.method_name << ", the method " << i
               << "th output phase tag " << input.first << " invalid";
      }
    }
  }
  return SUCCESS;
}

bool ServableSignature::GetMethodDeclare(const std::string &method_name, MethodSignature &method) {
  auto item =
    find_if(methods.begin(), methods.end(), [&](const MethodSignature &v) { return v.method_name == method_name; });
  if (item != methods.end()) {
    method = *item;
    return true;
  }
  return false;
}

void ServableStorage::Register(const ServableSignature &def) {
  auto model_name = def.servable_meta.servable_name;
  if (servable_signatures_map_.find(model_name) == servable_signatures_map_.end()) {
    MSI_LOG_WARNING << "Servable " << model_name << " has already been defined";
  }
  servable_signatures_map_[model_name] = def;
}

bool ServableStorage::GetServableDef(const std::string &model_name, ServableSignature &def) const {
  auto it = servable_signatures_map_.find(model_name);
  if (it == servable_signatures_map_.end()) {
    return false;
  }
  def = it->second;
  return true;
}

std::shared_ptr<ServableStorage> ServableStorage::Instance() {
  static std::shared_ptr<ServableStorage> storage;
  if (storage == nullptr) {
    storage = std::make_shared<ServableStorage>();
  }
  return storage;
}

void ServableStorage::RegisterMethod(const MethodSignature &method) {
  MSI_LOG_INFO << "Declare method " << method.method_name << ", servable " << method.servable_name;
  auto it = servable_signatures_map_.find(method.servable_name);
  if (it == servable_signatures_map_.end()) {
    ServableSignature signature;
    signature.methods.push_back(method);
    servable_signatures_map_[method.servable_name] = signature;
    return;
  }
  it->second.methods.push_back(method);
}

void ServableStorage::DeclareServable(const mindspore::serving::ServableMeta &servable) {
  MSI_LOG_INFO << "Declare servable " << servable.servable_name;
  auto it = servable_signatures_map_.find(servable.servable_name);
  if (it == servable_signatures_map_.end()) {
    ServableSignature signature;
    signature.servable_meta = servable;
    servable_signatures_map_[servable.servable_name] = signature;
    return;
  }
  it->second.servable_meta = servable;
}

void ServableStorage::RegisterInputOutputInfo(const std::string &servable_name, size_t inputs_count,
                                              size_t outputs_count) {
  auto it = servable_signatures_map_.find(servable_name);
  if (it == servable_signatures_map_.end()) {
    MSI_LOG_EXCEPTION << "RegisterInputOutputInfo failed, cannot find servable " << servable_name;
  }
  auto &servable_meta = it->second.servable_meta;
  if (servable_meta.inputs_count != 0 && servable_meta.inputs_count != inputs_count) {
    MSI_LOG_EXCEPTION << "RegisterInputOutputInfo failed, inputs count " << inputs_count << " not match old count "
                      << servable_meta.inputs_count << ",servable name " << servable_name;
  }
  if (servable_meta.outputs_count != 0 && servable_meta.outputs_count != outputs_count) {
    MSI_LOG_EXCEPTION << "RegisterInputOutputInfo failed, outputs count " << outputs_count << " not match old count "
                      << servable_meta.outputs_count << ",servable name " << servable_name;
  }
  servable_meta.inputs_count = inputs_count;
  servable_meta.outputs_count = outputs_count;
}

std::vector<size_t> ServableStorage::GetInputOutputInfo(const std::string &servable_name) const {
  std::vector<size_t> result;
  auto it = servable_signatures_map_.find(servable_name);
  if (it == servable_signatures_map_.end()) {
    return result;
  }
  result.push_back(it->second.servable_meta.inputs_count);
  result.push_back(it->second.servable_meta.outputs_count);
  return result;
}

}  // namespace mindspore::serving
