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

#include "worker/work_executor.h"
#include <utility>
#include <cstring>
#include <thread>
#include <chrono>
#include <map>
#include "worker/stage_function.h"
#include "common/tensor.h"
#include "worker/servable_register.h"

namespace mindspore::serving {

WorkExecutor::WorkExecutor() = default;

WorkExecutor::~WorkExecutor() { Stop(); }

Status WorkExecutor::Init(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &model_loaders) {
  Status status;
  if (init_flag_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Worker service has been initialized";
  }
  // servable can be nullptr
  model_loaders_ = model_loaders;
  status = ServableRegister::Instance().InitOnModelsLoad(model_loaders);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Init on models load failed";
    return status;
  }
  InitStageFunctionQueue();
  InitPredictTaskQueue();

  init_flag_ = true;
  return SUCCESS;
}

void WorkExecutor::StageCallback(const std::vector<InstancePtr> &instances,
                                 const std::vector<ResultInstance> &outputs) {
  if (instances.empty() || instances.size() != outputs.size()) {
    MSI_LOG_ERROR << "Invalid inputs size " << instances.size() << ", result size " << outputs.size();
    return;
  }
  // <method name, <stage index, instances>>
  std::map<std::string, std::map<uint64_t, std::vector<InstancePtr>>> outputs_real;
  for (size_t i = 0; i < instances.size(); i++) {
    auto &instance = instances[i];
    auto &output = outputs[i];
    if (output.error_msg != SUCCESS) {
      ReplyError(instance, output.error_msg);
      continue;
    }
    CreateResultInstance(instance, output);
    outputs_real[instance->method_def->method_name][instance->stage_index].push_back(instance);
  }
  for (auto &method_instances_it : outputs_real) {
    for (auto &stage_instances_it : method_instances_it.second) {
      auto &stage_instances = stage_instances_it.second;
      if (!stage_instances.empty()) {
        auto &method_def = *stage_instances[0]->method_def;
        auto stage_index = stage_instances_it.first;
        OnReceiveStageInputs(method_def, stage_index + 1, stage_instances);
      }
    }
  }
}

void WorkExecutor::InitStageFunctionQueue() {
  // init cpp preprocess and postprocess
  auto stage_callback = [this](const std::vector<InstancePtr> &instances, const std::vector<ResultInstance> &outputs) {
    StageCallback(instances, outputs);
  };
  auto const &signature = ServableRegister::Instance().GetServableSignature();
  // start task queue for handle preprocess and postprocess
  std::vector<MethodStage> py_stage_infos;
  std::vector<MethodStage> cpp_stage_infos;
  for (auto &method : signature.methods) {
    for (auto &stage_it : method.stage_map) {
      auto &stage = stage_it.second;
      if (stage.stage_type == kMethodStageTypePyFunction) {
        MSI_LOG_INFO << "PyFunction stage " << stage.stage_key << ", method name: " << stage.method_name
                     << ", stage index: " << stage.stage_index << ", batch size: " << stage.batch_size;
        py_stage_infos.push_back(stage);
      } else if (stage.stage_type == kMethodStageTypeCppFunction) {
        MSI_LOG_INFO << "CppFunction stage " << stage.stage_key << ", method name: " << stage.method_name
                     << ", stage index: " << stage.stage_index << ", batch size: " << stage.batch_size;
        cpp_stage_infos.push_back(stage);
      }
    }
  }
  if (!py_stage_infos.empty()) {
    py_task_queue_.Start("PyTask", py_stage_infos, stage_callback);
  }
  if (!cpp_stage_infos.empty()) {
    cpp_task_queue_pool_.Start("CppTask", cpp_stage_infos, stage_callback, 3);  // 3 thread
  }
}

void WorkExecutor::InitPredictTaskQueue() {
  auto stage_callback = [this](const std::vector<InstancePtr> &instances, const std::vector<ResultInstance> &outputs) {
    StageCallback(instances, outputs);
  };
  auto const &signature = ServableRegister::Instance().GetServableSignature();
  for (auto &model_meta : signature.model_metas) {
    auto model_key = model_meta.common_meta.model_key;
    auto &thread = predict_thread_map_[model_key];  // insert
    thread.Start("PredictTask", model_loaders_[model_key], model_meta, stage_callback);
  }
}

void WorkExecutor::Stop() {
  init_flag_ = false;
  for (auto &item : predict_thread_map_) {
    item.second.Stop();
  }
  predict_thread_map_.clear();
  ClearInstances(Status(WORKER_UNAVAILABLE, "Servable stopped"));
  for (auto &model : model_loaders_) {
    model.second->Clear();
  }
  model_loaders_.clear();
  py_task_queue_.Stop();
  cpp_task_queue_pool_.Stop();
}

Status WorkExecutor::Work(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                          const WorkCallBack &on_process_done) {
  if (!init_flag_) {
    MSI_LOG_EXCEPTION << "Worker service has not been initialized";
  }
  auto user_id = WorkExecutor::GetNextUserId();
  InferSession infer_session;
  infer_session.call_back = on_process_done;

  Status status;
  auto const &signature = ServableRegister::Instance().GetServableSignature();
  auto method_def = signature.GetMethodDeclare(request_spec.method_name);
  if (method_def == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Not support method " << request_spec.method_name;
  }

  std::vector<InstancePtr> instances;
  for (size_t i = 0; i < instances_data.size(); i++) {
    if (method_def->inputs.size() != instances_data[i].size()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "The inputs count " << instances_data[i].size() << " of instance " << i
                                            << " is not equal to the inputs count " << method_def->inputs.size()
                                            << " of the method " << request_spec.method_name;
    }

    auto instance = std::make_shared<Instance>();
    instances.push_back(instance);

    instance->method_def = method_def;
    instance->stage_data_list[0] = instances_data[i];  // stage 0 data: input
    instance->stage_max = method_def->GetStageMax();
    instance->user_id = user_id;
  }
  infer_session.instances = instances;
  {
    std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
    infer_session_map_[user_id] = infer_session;
  }
  OnReceiveStageInputs(*method_def, kStageStartIndex, instances);  // stage 1 is the first stage
  return SUCCESS;
}

void WorkExecutor::OnReceiveStageInputs(const MethodSignature &method_def, uint64_t stage_index,
                                        const std::vector<InstancePtr> &instances) {
  if (instances.empty()) {
    MSI_LOG_EXCEPTION << "Inputs cannot be empty";
  }
  auto stage_it = method_def.stage_map.find(stage_index);
  if (stage_it == method_def.stage_map.end()) {
    MSI_LOG_EXCEPTION << "Cannot find stage " << stage_index;
  }
  auto &stage = stage_it->second;
  CreateInputInstance(stage, instances);
  if (stage_index >= method_def.GetStageMax()) {
    ReplyRequest(instances);
    return;
  }
  if (stage.stage_type == kMethodStageTypePyFunction) {
    py_task_queue_.PushTask(method_def.method_name, stage_index, instances);
  } else if (stage.stage_type == kMethodStageTypeCppFunction) {
    cpp_task_queue_pool_.PushTask(method_def.method_name, stage_index, instances);
  } else if (stage.stage_type == kMethodStageTypeModel) {
    auto it = predict_thread_map_.find(stage.stage_key);
    if (it == predict_thread_map_.end()) {
      MSI_LOG_EXCEPTION << "Cannot find model " << stage.stage_key << " in predict_thread_map_";
    }
    it->second.PushPredictTask(stage, instances);
  } else {
    MSI_LOG_EXCEPTION << "Invalid stage type " << static_cast<int>(stage.stage_type);
  }
}

bool WorkExecutor::ReplyRequest(const std::vector<InstancePtr> &outputs) {
  MSI_TIME_STAMP_START(ReplyRequest)
  for (auto &item : outputs) {
    ReplyRequest(item);
  }
  MSI_TIME_STAMP_END(ReplyRequest)
  return true;
}

bool WorkExecutor::ReplyRequest(const InstancePtr &instance) {
  Status status;
  instance->error_msg = SUCCESS;
  instance->stage_data_list.clear();
  instance->stage_index = instance->stage_max;

  std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
  auto it = infer_session_map_.find(instance->user_id);
  if (it == infer_session_map_.end()) {
    MSI_LOG_WARNING << "Cannot find user in session map, user id " << instance->user_id;
    return false;
  }
  auto &infer_session = it->second;
  infer_session.reply_count++;
  if (infer_session.reply_count == infer_session.instances.size()) {
    infer_session.call_back(infer_session.instances);
    infer_session_map_.erase(it);
  }
  return true;
}

bool WorkExecutor::ReplyError(const InstancePtr &instance, const Status &error_msg) {
  instance->error_msg = error_msg;
  instance->data.clear();
  instance->stage_data_list.clear();
  instance->stage_index = instance->stage_max;

  std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
  auto it = infer_session_map_.find(instance->user_id);
  if (it == infer_session_map_.end()) {
    MSI_LOG_WARNING << "Cannot find user in session map, user id " << instance->user_id;
    return false;
  }
  auto &infer_session = it->second;
  infer_session.reply_count++;
  if (infer_session.reply_count == infer_session.instances.size()) {
    infer_session.call_back(infer_session.instances);
    infer_session_map_.erase(it);
  }
  return true;
}

void WorkExecutor::CreateInputInstance(const MethodStage &stage, const std::vector<InstancePtr> &instances) {
  for (auto &instance : instances) {
    CreateInputInstance(stage, instance);
  }
}

void WorkExecutor::CreateInputInstance(const MethodStage &stage, const InstancePtr &instance) {
  instance->data.clear();
  const auto &inputs = stage.stage_inputs;
  instance->stage_index = stage.stage_index;
  for (auto &item : inputs) {
    if (item.first >= instance->stage_data_list.size()) {
      MSI_LOG_EXCEPTION << "Invalid input stage index " << item.first << ", data stage count "
                        << instance->stage_data_list.size();
    }
    auto &data = instance->stage_data_list[item.first];
    if (data.size() <= item.second) {
      MSI_LOG_EXCEPTION << "Invalid output index " << item.second << ", output count " << data.size()
                        << ", input stage index " << item.first << ", stage index " << stage.stage_index << ", method "
                        << stage.method_name;
    }
    instance->data.push_back(data[item.second]);
  }
}

void WorkExecutor::CreateResultInstance(const InstancePtr &instance, const ResultInstance &result) {
  instance->data.clear();
  auto stage_index = instance->stage_index;
  instance->stage_data_list[stage_index] = result.data;
}

uint64_t WorkExecutor::GetNextUserId() {
  static std::atomic<uint64_t> user_id;
  return ++user_id;
}

uint64_t WorkExecutor::GetMaxBatchSize() const {
  uint64_t batch_size = 1;
  for (auto &model : predict_thread_map_) {
    auto model_batch = model.second.GetBatchSize();
    if (model_batch > batch_size) {
      batch_size = model_batch;
    }
  }
  return batch_size;
}

void WorkExecutor::ClearInstances(const Status &error_msg) {
  std::unique_lock<std::mutex> lock(infer_session_map_mutex_);
  MSI_LOG_INFO << "Clear instances, remain request count " << infer_session_map_.size();
  for (auto &item : infer_session_map_) {
    auto &infer_session = item.second;
    for (auto &instance : infer_session.instances) {
      if (instance->stage_index != instance->stage_max) {
        instance->error_msg = error_msg;
      }
    }
    item.second.call_back(item.second.instances);
  }
  infer_session_map_.clear();
}

}  // namespace mindspore::serving
