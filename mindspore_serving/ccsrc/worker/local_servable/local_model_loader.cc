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

#include "worker/local_servable/local_model_loader.h"
#include <vector>
#include <string>
#include "common/tensor.h"
#include "worker/context.h"
#include "worker/servable_register.h"

namespace mindspore::serving {

LocalModelLoader::~LocalModelLoader() { Clear(); }

uint64_t LocalModelLoader::GetGraphNum() const {
  if (!model_session_) {
    MSI_LOG_EXCEPTION << "Model '" << GetModelKey() << "' has not been loaded";
  }
  return graph_num_;
}

Status LocalModelLoader::Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                                 uint64_t subgraph) {
  if (!model_session_) {
    MSI_LOG_EXCEPTION << "Model '" << GetModelKey() << "' has not been loaded";
  }
  return model_session_->ExecuteModel(input, output, true, subgraph);
}

std::vector<TensorInfo> LocalModelLoader::GetInputInfos(uint64_t subgraph) const {
  if (!model_session_) {
    MSI_LOG_EXCEPTION << "Model '" << GetModelKey() << "' has not been loaded";
  }
  return model_session_->GetInputInfos(subgraph);
}

std::vector<TensorInfo> LocalModelLoader::GetOutputInfos(uint64_t subgraph) const {
  if (!model_session_) {
    MSI_LOG_EXCEPTION << "Model '" << GetModelKey() << "' has not been loaded";
  }
  return model_session_->GetOutputInfos(subgraph);
}

uint64_t LocalModelLoader::GetBatchSize() const {
  if (!model_session_) {
    MSI_LOG_EXCEPTION << "Model '" << GetModelKey() << "' has not been loaded";
  }
  return model_session_->GetBatchSize();
}

Status LocalModelLoader::LoadModel(const std::string &servable_directory, const std::string &servable_name,
                                   uint64_t version_number, const ModelMeta &model_meta, const std::string &dec_key,
                                   const std::string &dec_mode) {
  if (model_loaded_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Model has loaded";
  }
  base_spec_.servable_directory = servable_directory;
  base_spec_.servable_name = servable_name;
  base_spec_.version_number = version_number;
  model_meta_ = model_meta;

  Status status;
  const ServableSignature &signature = ServableRegister::Instance().GetServableSignature();
  if (signature.servable_name != servable_name) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable '" << servable_name << "' has not been registered";
  }
  if (signature.servable_type != kServableTypeLocal) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable '" << servable_name << "' is not registered as local servable";
  }
  status = InitDevice(model_meta.local_meta.model_format);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Init env failed";
    return status;
  }
  status = LoadModel(version_number, dec_key, dec_mode);
  if (status != SUCCESS) {
    return status;
  }
  model_loaded_ = true;
  return SUCCESS;
}

Status LocalModelLoader::InitDevice(ModelType model_type) {
  Status status;
  auto context = ServableContext::Instance();
  auto device_type = context->GetDeviceType();
  auto lite_backend = InferenceLoader::Instance().GetEnableLite();
  auto support_device_type = InferenceLoader::Instance().GetSupportDeviceType(device_type, model_type);
  if (support_device_type == kDeviceTypeNotSpecified || (lite_backend && model_type != kMindIR_Opt)) {
    std::string inference_package = lite_backend ? "MindSpore Lite" : "MindSpore";
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Not support device type " << device_type << " and model type " << model_type
           << ". Current inference backend: " << inference_package
           << ". When the inference backend is MindSpore, Ascend 910/710/310 and GPU supports MindIR "
           << "model, and Ascend 710/310 supports OM model. When the inference backend is MindSpore Lite, "
           << "Ascend 710/310, GPU and CPU only support MindIR_Opt model converted by Lite converter tool.";
  }
  context->SetDeviceType(support_device_type);
  return SUCCESS;
}

Status LocalModelLoader::LoadModel(uint64_t version_number, const std::string &dec_key, const std::string &dec_mode) {
  const auto &model_meta = model_meta_;
  auto context = ServableContext::Instance();
  std::string model_dir =
    base_spec_.servable_directory + "/" + base_spec_.servable_name + "/" + std::to_string(version_number);
  if (!common::DirOrFileExist(model_dir)) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Start servable failed: There is no specified version directory of models, specified version number: "
           << version_number << ", servable directory: '" << base_spec_.servable_directory << "', servable name: '"
           << base_spec_.servable_name << "'";
  }
  const auto &common_meta = model_meta.common_meta;
  const auto &local_meta = model_meta.local_meta;
  std::vector<std::string> model_file_names;
  for (auto &file : local_meta.model_files) {
    std::string model_file_name = model_dir + "/" + file;
    model_file_names.push_back(model_file_name);
  }
  auto session = InferenceLoader::Instance().CreateMindSporeInfer();
  if (session == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Create MindSpore infer failed";
  }
  std::string config_file_path;
  if (!local_meta.config_file.empty()) {
    if (local_meta.config_file[0] == '/') {
      config_file_path = local_meta.config_file;
    } else {
      config_file_path = base_spec_.servable_directory + "/" + base_spec_.servable_name + "/" + local_meta.config_file;
    }
  }
  auto enable_lite = InferenceLoader::Instance().GetEnableLite();
  Status status = session->LoadModelFromFile(context->GetDeviceType(), context->GetDeviceId(), model_file_names,
                                             local_meta.model_format, common_meta.with_batch_dim,
                                             common_meta.without_batch_dim_inputs, model_meta.local_meta.model_context,
                                             dec_key, dec_mode, config_file_path, enable_lite);
  if (status != SUCCESS) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Load model failed, servable directory: '" << base_spec_.servable_directory << "', servable name: '"
           << base_spec_.servable_name << "', model file: '" << local_meta.model_files << "', version number "
           << version_number << ",model context: " << local_meta.model_context.AsString()
           << ", load error details: " << status.StatusMessage();
  }
  model_session_ = session;
  graph_num_ = model_file_names.size();

  MSI_LOG_INFO << "Load model success, servable directory: '" << base_spec_.servable_directory << "', servable name: '"
               << base_spec_.servable_name << "', model file: '" << local_meta.model_files << "', version number "
               << version_number << ", context " << local_meta.model_context.AsString();
  return SUCCESS;
}

void LocalModelLoader::Clear() {
  if (model_session_ != nullptr) {
    model_session_->UnloadModel();
    model_session_ = nullptr;
  }
  model_loaded_ = false;
}

}  // namespace mindspore::serving
