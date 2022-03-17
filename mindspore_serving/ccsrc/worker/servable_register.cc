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
#include "worker/servable_register.h"
#include <set>
#include <string>
#include "worker/stage_function.h"

namespace mindspore {
namespace serving {
ServableRegister &ServableRegister::Instance() {
  static ServableRegister storage;
  return storage;
}

Status ServableRegister::RegisterMethod(const MethodSignature &method) {
  MSI_LOG_INFO << "Declare method " << method.method_name << ", servable " << method.servable_name;
  servable_signatures_.servable_name = method.servable_name;
  for (auto &item : servable_signatures_.methods) {
    // cppcheck-suppress useStlAlgorithm
    if (item.method_name == method.method_name) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Method " << method.method_name << " has been registered more than once.";
    }
  }
  servable_signatures_.methods.push_back(method);
  return SUCCESS;
}

Status ServableRegister::DeclareModel(ModelMeta model) {
  auto &common_meta = model.common_meta;
  auto &local_meta = model.local_meta;
  MSI_LOG_INFO << "Declare model " << local_meta.model_files;
  if (servable_signatures_.servable_type == kServableTypeDistributed) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare model failed, servable has already been declared as distributed servable";
  }
  servable_signatures_.servable_name = common_meta.servable_name;
  servable_signatures_.servable_type = kServableTypeLocal;
  if (local_meta.model_files.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Declare model failed, model files size cannot be 0";
  }
  std::set<std::string> cur_model_files;
  for (auto &model_item : servable_signatures_.model_metas) {
    for (auto &file_item : model_item.local_meta.model_files) {
      (void)cur_model_files.emplace(file_item);
    }
  }
  for (auto &file : local_meta.model_files) {
    if (file.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Declare model " << local_meta.model_files << " failed, model file cannot be empty";
    }
    if (cur_model_files.count(file) > 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Declare model " << local_meta.model_files << " failed, model file '"
                                            << file << "' has already been used";
    }
  }
  if (local_meta.model_format == ModelType::kUnknownType) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare model " << local_meta.model_files << " failed, model_format is not inited";
  }
  for (auto &item : servable_signatures_.model_metas) {
    // cppcheck-suppress useStlAlgorithm
    if (item.common_meta.model_key == common_meta.model_key) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Declare model " << local_meta.model_files << " failed, the same model has already been declared";
    }
  }
  servable_signatures_.model_metas.push_back(model);
  return SUCCESS;
}

Status ServableRegister::DeclareDistributedModel(ModelMeta model) {
  auto &common_meta = model.common_meta;
  MSI_LOG_INFO << "Declare distributed model " << common_meta.servable_name;
  if (servable_signatures_.servable_type == kServableTypeDistributed) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare distributed model failed, servable is repeatedly been declared as distributed servable";
  }
  if (servable_signatures_.servable_type == kServableTypeLocal) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare distributed model failed, servable has already been declared as local servable";
  }
  servable_signatures_.servable_name = common_meta.servable_name;
  servable_signatures_.servable_type = kServableTypeDistributed;
  if (model.distributed_meta.rank_size == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare distributed model " << common_meta.servable_name << " failed, rank_size cannot be 0";
  }
  if (model.distributed_meta.stage_size == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Declare distributed model " << common_meta.servable_name << " failed, stage_size cannot be 0";
  }
  servable_signatures_.model_metas.push_back(model);
  return SUCCESS;
}

Status ServableRegister::RegisterInputOutputInfo(const std::string &model_key, size_t inputs_count,
                                                 size_t outputs_count, uint64_t subgraph) {
  MSI_LOG_INFO << "Declare model " << model_key << " subgraph " << subgraph << " inputs count " << inputs_count
               << " outputs count " << outputs_count;
  auto &model_metas = servable_signatures_.model_metas;
  auto it = std::find_if(model_metas.begin(), model_metas.end(),
                         [model_key](const ModelMeta &item) { return item.common_meta.model_key == model_key; });
  if (it == model_metas.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "RegisterInputOutputInfo failed, cannot find model " << model_key;
  }
  auto &common_meta = it->common_meta;

  if (common_meta.inputs_count.count(subgraph) > 0 && common_meta.inputs_count[subgraph] != inputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "RegisterInputOutputInfo failed, inputs count " << inputs_count << " not match old count "
           << common_meta.inputs_count[subgraph] << ", model: " << model_key;
  }
  if (common_meta.outputs_count.count(subgraph) > 0 && common_meta.outputs_count[subgraph] != outputs_count) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "RegisterInputOutputInfo failed, outputs count " << outputs_count << " not match old count "
           << common_meta.outputs_count[subgraph] << ", model: " << model_key;
  }
  common_meta.inputs_count[subgraph] = inputs_count;
  common_meta.outputs_count[subgraph] = outputs_count;
  return SUCCESS;
}

Status ServableRegister::InitCallModelMethods(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models) {
  for (auto &model_it : models) {
    auto model_key = model_it.first;
    auto &model_loader = model_it.second;
    auto graph_num = model_loader->GetGraphNum();
    for (size_t subgraph = 0; subgraph < graph_num; subgraph++) {
      auto input_infos = model_loader->GetInputInfos(subgraph);
      auto output_infos = model_loader->GetOutputInfos(subgraph);
      auto status = RegisterOneCallModelMethod(model_key, input_infos.size(), output_infos.size(), subgraph);
      if (status != SUCCESS) {
        return status;
      }
    }
  }
  return SUCCESS;
}

std::string ServableRegister::GetCallModelMethodName(const std::string &model_key, uint64_t subgraph) {
  std::string method_name = "@call_" + model_key + "_" + std::to_string(subgraph);
  return method_name;
}

Status ServableRegister::RegisterOneCallModelMethod(const std::string &model_key, uint64_t input_count,
                                                    uint64_t output_count, uint64_t subgraph) {
  std::string method_name = GetCallModelMethodName(model_key, subgraph);
  MethodSignature method;
  method.method_name = method_name;
  method.servable_name = servable_signatures_.servable_name;

  std::vector<std::pair<size_t, uint64_t>> model_inputs;
  for (uint64_t i = 0; i < input_count; i++) {
    (void)method.inputs.emplace_back("x" + std::to_string(i));
    (void)model_inputs.emplace_back(0, i);  // all method inputs are function inputs
  }
  std::vector<std::pair<size_t, uint64_t>> returns;
  for (uint64_t i = 0; i < output_count; i++) {
    (void)method.outputs.emplace_back("y" + std::to_string(i));
    (void)returns.emplace_back(1, i);
  }
  method.AddStageModel(model_key, model_inputs, subgraph);
  method.SetReturn(returns);
  auto status = RegisterMethod(method);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register Method failed";
    return status;
  }
  status = RegisterInputOutputInfo(model_key, input_count, output_count, subgraph);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register model input and output info failed";
    return status;
  }
  return SUCCESS;
}

Status ServableRegister::CheckModels(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models) {
  auto const &signature = servable_signatures_;
  if (signature.methods.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "There is no method registered for servable";
  }
  if (models.size() != signature.model_metas.size()) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "The number " << signature.model_metas.size() << " of models declared is not equal to the number "
           << models.size() << " of models loaded";
  }
  for (auto &model_meta : signature.model_metas) {
    auto &model_key = model_meta.common_meta.model_key;
    auto model_load_it = models.find(model_key);
    if (model_load_it == models.end()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Model " << model_key << " has not been loaded";
    }
    auto &model_loader = model_load_it->second;
    auto batch_size = model_loader->GetBatchSize();
    if (batch_size == 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid batch size 0, model info: " << model_key;
    }
    auto graph_num = model_loader->GetGraphNum();
    if (graph_num == 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Invalid subgraph number 0, model info: " << model_key;
    }
    for (uint64_t subgraph = 0; subgraph < graph_num; subgraph++) {
      auto input_infos = model_loader->GetInputInfos(subgraph);
      auto output_infos = model_loader->GetOutputInfos(subgraph);

      MSI_LOG_INFO << "Print model info, model info: '" << model_meta.common_meta.model_key << "', subgraph "
                   << subgraph;
      MSI_LOG_INFO << "Model input infos: count " << input_infos.size();
      for (auto &item : input_infos) {
        MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
      }
      MSI_LOG_INFO << "Model output infos: count " << output_infos.size();
      for (auto &item : output_infos) {
        MSI_LOG_INFO << item.shape << ", " << item.data_type << ", " << item.size;
      }

      const auto &common_meta = model_meta.common_meta;
      if (common_meta.inputs_count.count(subgraph) > 0 && input_infos.size() != common_meta.inputs_count.at(subgraph)) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "The inputs count " << common_meta.inputs_count.at(subgraph) << " in register_method "
               << "not equal to the count " << input_infos.size() << " defined in model, model info: " << model_key
               << ", subgraph: " << subgraph;
      }
      if (common_meta.outputs_count.count(subgraph) > 0 &&
          output_infos.size() != common_meta.outputs_count.at(subgraph)) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "The outputs count " << common_meta.outputs_count.at(subgraph) << " in register_method "
               << "not equal to the count " << output_infos.size() << " defined in model, model info: " << model_key
               << ", subgraph: " << subgraph;
      }
    }
  }
  return SUCCESS;
}

Status ServableRegister::CheckOneMethod(const MethodSignature &method) {
  const auto &servable_name = servable_signatures_.servable_name;
  const auto &model_metas = servable_signatures_.model_metas;
  for (auto &stage_it : method.stage_map) {
    auto stage_index = stage_it.first;
    auto &stage = stage_it.second;
    for (size_t input_index = 0; input_index < stage.stage_inputs.size(); input_index++) {
      auto input_stage_index = stage.stage_inputs[input_index].first;
      auto output_index = stage.stage_inputs[input_index].second;
      // method input
      if (input_stage_index == 0) {
        if (output_index >= method.inputs.size()) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "The stage " << stage_index << " " << input_index << "th input uses method " << output_index
                 << "th input, that is greater than the method inputs size " << method.inputs.size()
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
        continue;
      }
      // check input stage index
      if (input_stage_index >= stage_index) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "The " << input_index << "th input data of stage " << stage_index << " cannot not come from stage "
               << input_stage_index << ", servable: " << servable_name << ", method: " << method.method_name;
      }
      // check input stage output index
      auto it = method.stage_map.find(input_stage_index);
      if (it == method.stage_map.end()) {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Cannot find stage " << input_stage_index << " from method define information, "
               << ", servable: " << servable_name << ", method: " << method.method_name;
      }
      const auto &input_stage = it->second;
      if (input_stage.stage_type == kMethodStageTypePyFunction) {
        size_t input_count, output_count;
        if (!PyStageFunctionStorage::Instance()->GetPyFunctionInfo(input_stage.stage_key, &input_count,
                                                                   &output_count)) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "PyFunction " << input_stage.stage_key << " is not defined, "
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
        if (output_index >= output_count) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "The stage(begin with 1) " << stage_index << " " << input_index << "th input uses python function "
                 << input_stage.stage_key << " " << output_index
                 << "th output, that is greater than the function output size " << output_count
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
      } else if (input_stage.stage_type == kMethodStageTypeCppFunction) {
        auto function = CppStageFunctionStorage::Instance().GetFunction(input_stage.stage_key);
        if (function == nullptr) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "CppFunction " << input_stage.stage_key << " is not defined, "
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
        auto func_output_count = function->GetOutputsCount(input_stage.stage_key);
        if (output_index >= func_output_count) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "The stage(begin with 1) " << stage_index << " " << input_index << "th input uses c++ function "
                 << input_stage.stage_key << " " << output_index
                 << "th output, that is greater than the function output size " << func_output_count
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
      } else if (input_stage.stage_type == kMethodStageTypeModel) {
        auto model_it =
          std::find_if(model_metas.begin(), model_metas.end(), [&input_stage](const ModelMeta &model_meta) {
            return input_stage.stage_key == model_meta.common_meta.model_key;
          });
        if (model_it == model_metas.end()) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "Model " << input_stage.stage_key << " is not defined, "
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
        if (model_it->common_meta.outputs_count.count(input_stage.subgraph) == 0) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "Model " << input_stage.stage_key << " subgraph " << input_stage.subgraph << " is not declared"
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
        auto model_output_count = model_it->common_meta.outputs_count.at(input_stage.subgraph);
        if (output_index >= model_output_count) {
          return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                 << "The stage(begin with 1) " << stage_index << " " << input_index << "th input uses model "
                 << input_stage.stage_key << " subgraph " << input_stage.subgraph << " " << output_index
                 << "th output, that is greater than the model output size " << model_output_count
                 << ", servable: " << servable_name << ", method: " << method.method_name;
        }
      } else {
        return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
               << "Invalid stage type " << static_cast<int>(stage.stage_type) << ", servable: " << servable_name
               << ", method: " << method.method_name;
      }
    }
  }
  return SUCCESS;
}

Status ServableRegister::CheckMethods() {
  std::set<std::string> method_set;
  Status status;
  for (const auto &method : servable_signatures_.methods) {
    if (method_set.count(method.method_name) > 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Servable " << servable_signatures_.servable_name << " method '"
                                            << method.method_name << "' has been defined repeatedly";
    }
    (void)method_set.emplace(method.method_name);
    status = CheckOneMethod(method);
    if (status != SUCCESS) {
      return status;
    }
  }
  return SUCCESS;
}

Status ServableRegister::InitMethodBatchSize(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models) {
  // stages only use method inputs as inputs batch_size == mini model batch size
  // other stages batch_size == max model batch size
  for (auto &method : servable_signatures_.methods) {
    uint64_t mini_batch_size = UINT32_MAX;
    uint64_t max_batch_size = 0;
    for (auto &stage_it : method.stage_map) {
      auto &stage = stage_it.second;
      if (stage.stage_type == kMethodStageTypeModel) {
        auto model_it = models.find(stage.stage_key);
        if (model_it == models.end()) {
          return INFER_STATUS_LOG_ERROR(FAILED) << "Model " << stage.stage_key << " has not been loaded";
        }
        stage.batch_size = model_it->second->GetBatchSize();
        if (stage.batch_size < mini_batch_size) {
          mini_batch_size = stage.batch_size;
        }
        if (stage.batch_size > max_batch_size) {
          max_batch_size = stage.batch_size;
        }
      }
    }
    if (mini_batch_size == UINT32_MAX || max_batch_size == 0) {
      mini_batch_size = 1;
      max_batch_size = 1;
    }
    for (auto &stage_it : method.stage_map) {
      auto &stage = stage_it.second;
      if (stage.stage_type != kMethodStageTypeModel && stage.batch_size == 0) {
        auto all_method_input = std::all_of(stage.stage_inputs.begin(), stage.stage_inputs.end(),
                                            [](const std::pair<size_t, uint64_t> &item) { return item.first == 0; });
        if (all_method_input) {
          stage.batch_size = mini_batch_size;
        } else {
          stage.batch_size = max_batch_size;
        }
      }
    }
  }
  return SUCCESS;
}

Status ServableRegister::InitOnModelsLoad(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models) {
  Status status;
  status = CheckModels(models);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check models failed";
    return status;
  }
  status = InitCallModelMethods(models);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Init call model methods failed";
    return status;
  }
  status = CheckMethods();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check methods failed";
    return status;
  }
  status = InitMethodBatchSize(models);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Init models batch size failed";
    return status;
  }
  return SUCCESS;
}
}  // namespace serving
}  // namespace mindspore
