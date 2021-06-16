/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "worker/local_servable/local_sevable.h"
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <string>
#include "common/tensor.h"
#include "common/file_system_operation.h"
#include "worker/context.h"

namespace {
static const char *kVersionStrategyLatest = "latest";
static const char *kVersionStrategySpecific = "specific";
}  // namespace

namespace mindspore::serving {

LocalModelServable::~LocalModelServable() { Clear(); }

std::string LocalModelServable::GetServableDirectory() const { return base_spec_.servable_directory; }

std::string LocalModelServable::GetServableName() const { return servable_name_; }

uint64_t LocalModelServable::GetServableVersion() const { return running_version_number_; }

uint64_t LocalModelServable::GetConfigVersion() const { return base_spec_.version_number; }

uint64_t LocalModelServable::GetGraphNum() const { return graph_num_; }

Status LocalModelServable::Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                                   uint64_t subgraph) {
  if (!model_loaded_ || !session_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return session_->ExecuteModel(input, output, true, subgraph);
}

std::vector<TensorInfo> LocalModelServable::GetInputInfos(uint64_t subgraph) const {
  if (!model_loaded_ || !session_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return session_->GetInputInfos();
}

std::vector<TensorInfo> LocalModelServable::GetOutputInfos(uint64_t subgraph) const {
  if (!model_loaded_ || !session_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return session_->GetOutputInfos();
}

uint64_t LocalModelServable::GetBatchSize(uint64_t subgraph) const {
  if (!model_loaded_ || !session_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return session_->GetBatchSize();
}

Status LocalModelServable::StartServable(const std::string &servable_directory, const std::string &servable_name,
                                         uint64_t version_number, const std::string &dec_key,
                                         const std::string &dec_mode) {
  if (model_loaded_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Model has loaded";
  }
  session_ = InferenceLoader::Instance().CreateMindSporeInfer();
  if (session_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Create MindSpore infer failed";
  }

  base_spec_.servable_directory = servable_directory;
  base_spec_.servable_name = servable_name;
  base_spec_.version_number = version_number;

  std::string version_strategy;
  if (version_number == 0) {
    version_strategy = kVersionStrategyLatest;
  } else {
    version_strategy = kVersionStrategySpecific;
  }
  Status status;
  ServableSignature signature;
  if (!ServableStorage::Instance().GetServableDef(servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable '" << servable_name << "' has not been registered";
  }
  status = InitDevice(signature.servable_meta.local_meta.model_format, {});
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Init env failed";
    return status;
  }

  std::vector<uint64_t> real_versions;
  status = LoadServableConfig(base_spec_, version_strategy, &real_versions);
  if (status != SUCCESS) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Start servable failed, there is no servable of the specified version number, specified version number: "
           << version_number << ", servable directory: '" << base_spec_.servable_directory << "', servable name: '"
           << base_spec_.servable_name
           << "'. version number is a positive integer(started from 1) and 0 represents the maximum version number.";
  }
  auto real_version_number = real_versions[0];
  status = LoadModel(real_version_number, dec_key, dec_mode);
  if (status != SUCCESS) {
    return status;
  }
  servable_name_ = base_spec_.servable_name;
  running_version_number_ = real_version_number;
  model_loaded_ = true;
  graph_num_ = 1;
  MSI_LOG_INFO << status.StatusMessage();
  std::cout << status.StatusMessage() << std::endl;
  return SUCCESS;
}

void LocalModelServable::GetVersions(const ServableLoadSpec &servable_spec, std::vector<uint64_t> *real_versions) {
  MSI_EXCEPTION_IF_NULL(real_versions);
  // define version_strategy:"specific","latest","multi"
  if (version_strategy_ == kVersionStrategySpecific) {
    real_versions->push_back(servable_spec.version_number);
    return;
  }
  auto trans_to_integer = [](const std::string &str) -> uint32_t {
    uint32_t parsed_value = 0;
    for (auto c : str) {
      if (c < '0' || c > '9') {
        return 0;
      }
      parsed_value = parsed_value * 10 + c - '0';
    }
    if (std::to_string(parsed_value) != str) {
      return 0;
    }
    return parsed_value;
  };
  uint64_t newest_version = 0;
  std::string model_path = servable_spec.servable_directory + "/" + servable_spec.servable_name;
  auto sub_dir = GetAllSubDirsNotFullPath(model_path);
  static std::set<std::string> ignore_dir;
  for (const auto &dir : sub_dir) {
    if (dir == "__pycache__") continue;
    auto version_parse = trans_to_integer(dir);
    if (version_parse == 0) {
      if (ignore_dir.emplace(servable_spec.servable_directory + dir).second) {
        MSI_LOG_INFO << "Ignore directory " << dir << ", model_directory " << servable_spec.servable_directory
                     << ", model_name " << servable_spec.servable_name;
      }
      continue;
    }
    real_versions->push_back(version_parse);
    if (version_parse > newest_version) {
      newest_version = version_parse;
    }
  }
  if (version_strategy_ == kVersionStrategyLatest) {
    real_versions->clear();
    if (newest_version != 0) {
      real_versions->push_back(newest_version);
    }
  }
}

Status LocalModelServable::LoadServableConfig(const ServableLoadSpec &servable_spec,
                                              const std::string &version_strategy,
                                              std::vector<uint64_t> *real_versions) {
  MSI_EXCEPTION_IF_NULL(real_versions);
  auto model_directory = servable_spec.servable_directory;
  auto model_name = servable_spec.servable_name;

  if (!DirOrFileExist(model_directory + "/" + model_name)) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Model not found, model_directory " << model_directory << ", model_name " << model_name;
  }
  std::string model_path = model_directory + "/" + model_name;
  auto version_directory = [model_path](int64_t version_number) {
    return model_path + "/" + std::to_string(version_number);
  };
  version_strategy_ = version_strategy;
  // version_strategy:"specific","latest","multi"
  GetVersions(servable_spec, real_versions);
  if (real_versions->size() == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Not found invalid model version , model_directory " << model_directory << ", model_name " << model_name;
  }
  for (auto real_version_number : *real_versions) {
    if (!DirOrFileExist(version_directory(real_version_number))) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Open failed for version " << real_version_number << ", model_directory "
                                            << model_directory << ", model_name " << model_name;
    }
  }
  return SUCCESS;
}

Status LocalModelServable::InitDevice(ModelType model_type, const std::map<std::string, std::string> &other_options) {
  Status status;
  auto context = ServableContext::Instance();
  DeviceType device_type = ServableContext::Instance()->GetDeviceType();
  auto support_device_type = InferenceLoader::Instance().GetSupportDeviceType(device_type, model_type);
  if (support_device_type == kDeviceTypeNotSpecified) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Not support device type " << device_type << " and model type " << model_type
           << ". Ascend 910, Ascend 310 and GPU supports MindIR model, and Ascend 310 supports OM model";
  }
  context->SetDeviceType(support_device_type);
  return SUCCESS;
}

Status LocalModelServable::LoadModel(uint64_t version_number, const std::string &dec_key, const std::string &dec_mode) {
  ServableSignature signature;
  if (!ServableStorage::Instance().GetServableDef(base_spec_.servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable " << base_spec_.servable_name << " has not been registered";
  }
  const auto &servable_meta = signature.servable_meta;
  const auto &common_meta = servable_meta.common_meta;
  const auto &local_meta = servable_meta.local_meta;
  std::string model_file_name = base_spec_.servable_directory + "/" + base_spec_.servable_name + "/" +
                                std::to_string(version_number) + "/" + local_meta.servable_file;
  auto context = ServableContext::Instance();
  Status status = session_->LoadModelFromFile(
    context->GetDeviceType(), context->GetDeviceId(), {model_file_name}, local_meta.model_format,
    common_meta.with_batch_dim, common_meta.without_batch_dim_inputs, local_meta.load_options, dec_key, dec_mode);
  if (status != SUCCESS) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Load model failed, servable directory: '" << base_spec_.servable_directory << "', servable name: '"
           << base_spec_.servable_name << "', servable file: '" << local_meta.servable_file << "', version number "
           << version_number << ", options " << local_meta.load_options;
  }
  return SUCCESS;
}

void LocalModelServable::Clear() {
  if (model_loaded_) {
    session_->UnloadModel();
    session_ = nullptr;
  }
  model_loaded_ = false;
}

}  // namespace mindspore::serving
