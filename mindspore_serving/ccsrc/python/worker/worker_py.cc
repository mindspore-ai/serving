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

#include "python/worker/worker_py.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "common/exit_handle.h"
#include "worker/notfiy_master/grpc_notify.h"
#include "worker/local_servable/local_model_loader.h"
#include "worker/distributed_worker/distributed_model_loader.h"
#include "worker/inference/inference.h"
#include "worker/servable_register.h"
#include "worker/extra_worker/remote_call_model.h"
#include "worker/context.h"

namespace mindspore::serving {
void PyWorker::StartServable(const std::string &servable_directory, const std::string &servable_name,
                             uint32_t version_number, const std::string &master_address,
                             const std::string &worker_address, const std::string &dec_key,
                             const std::string &dec_mode) {
  if (Worker::GetInstance().IsRunning()) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }
  const auto &signature = ServableRegister::Instance().GetServableSignature();
  if (signature.servable_name != servable_name) {
    MSI_LOG_EXCEPTION << "Servable '" << servable_name << "' has not been registered";
  }
  Status status;
  std::map<std::string, std::shared_ptr<ModelLoaderBase>> models_loader;
  status =
    LoadLocalModels(servable_directory, servable_name, version_number, dec_key, dec_mode, signature, &models_loader);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = Worker::GetInstance().StartServable(servable_directory, servable_name, version_number, models_loader,
                                               master_address, worker_address, true);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

Status PyWorker::LoadLocalModels(const std::string &servable_directory, const std::string &servable_name,
                                 uint32_t version_number, const std::string &dec_key, const std::string &dec_mode,
                                 const ServableSignature &signature,
                                 std::map<std::string, std::shared_ptr<ModelLoaderBase>> *models_loader) {
  Status status;
  for (auto &model_meta : signature.model_metas) {
    auto &model_key = model_meta.common_meta.model_key;
    auto local_models_loader = std::make_shared<LocalModelLoader>();
    status =
      local_models_loader->LoadModel(servable_directory, servable_name, version_number, model_meta, dec_key, dec_mode);
    if (status != SUCCESS) {
      local_models_loader->Clear();
      return status;
    }
    status = local_models_loader->AfterLoadModel();
    if (status != SUCCESS) {
      local_models_loader->Clear();
      return status;
    }
    (void)models_loader->emplace(model_key, local_models_loader);
  }
  return SUCCESS;
}

void PyWorker::StartDistributedServable(const std::string &servable_directory, const std::string &servable_name,
                                        const std::string &rank_table_json_file, uint32_t version_number,
                                        const std::string &distributed_address, const std::string &master_address,
                                        const std::string &worker_address, uint32_t wait_agents_time_in_seconds) {
  if (Worker::GetInstance().IsRunning()) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }

  Status status;
  auto model_loader = std::make_shared<DistributedModelLoader>();
  status = Worker::GetInstance().StartDistributedGrpcServer(model_loader, distributed_address);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }

  status = model_loader->LoadModel(servable_name, rank_table_json_file, wait_agents_time_in_seconds);
  if (status != SUCCESS) {
    model_loader->Clear();
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = model_loader->AfterLoadModel();
  if (status != SUCCESS) {
    model_loader->Clear();
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  std::map<std::string, std::shared_ptr<ModelLoaderBase>> models_loader;
  models_loader[model_loader->GetModelKey()] = model_loader;
  status = Worker::GetInstance().StartServable(servable_directory, servable_name, version_number, models_loader,
                                               master_address, worker_address, true);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

void PyWorker::StartExtraServable(const std::string &servable_directory, const std::string &servable_name,
                                  uint32_t version_number, bool device_ids_empty, const std::string &dec_key,
                                  const std::string &dec_mode, const std::string &master_address,
                                  const std::string &worker_address) {
  if (Worker::GetInstance().IsRunning()) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }
  const auto &signature = ServableRegister::Instance().GetServableSignature();
  if (signature.servable_name != servable_name) {
    MSI_LOG_EXCEPTION << "Servable '" << servable_name << "' has not been registered";
  }
  auto own_device = false;
  std::map<std::string, std::shared_ptr<ModelLoaderBase>> model_loaders;
  Status status;
  if (!signature.model_metas.empty()) {
    // if device_type is None, device_ids is empty, and there are models declared, Cpu target should be support
    auto target_device_type = ServableContext::Instance()->GetDeviceType();
    if (target_device_type == kDeviceTypeNotSpecified && device_ids_empty) {
      auto support_device_type = InferenceLoader::Instance().GetSupportDeviceType(kDeviceTypeCpu, kUnknownType);
      if (support_device_type == kDeviceTypeNotSpecified) {
        MSI_LOG_EXCEPTION
          << "Servable '" << servable_name << "' has models declared by declare_model, but parameter 'device_ids'"
          << " of ServableStartConfig is not set in Serving startup script when the MindSpore or Lite inference"
          << " package not support CPU";
      }
      target_device_type = kDeviceTypeCpu;
      ServableContext::Instance()->SetDeviceType(target_device_type);
    }
    if (target_device_type == kDeviceTypeCpu) {
      own_device = true;
      status = LoadLocalModels(servable_directory, servable_name, version_number, dec_key, dec_mode, signature,
                               &model_loaders);
      if (status != SUCCESS) {
        MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
      }
    } else {
      status = RemoteCallModel::InitRemote(servable_name, version_number, master_address, &model_loaders);
      if (status != SUCCESS) {
        MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
      }
    }
  }
  status = Worker::GetInstance().StartServable(servable_directory, servable_name, version_number, model_loaders,
                                               master_address, worker_address, own_device);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

std::vector<std::string> PyWorker::GetDeclaredModelNames() {
  std::vector<std::string> model_names;
  for (auto &model_meta : ServableRegister::Instance().GetServableSignature().model_metas) {
    // cppcheck-suppress useStlAlgorithm
    model_names.push_back(model_meta.common_meta.model_key);
  }
  return model_names;
}

bool PyWorker::EnablePyTaskQueue() { return Worker::GetInstance().GetWorkExecutor().GetPyTaskQueue().IsRunning(); }

TaskItem PyWorker::GetPyTask() {
  TaskItem item;
  Worker::GetInstance().GetWorkExecutor().GetPyTaskQueue().PyPopTask(&item);
  return item;
}

void PyWorker::PushPyTaskResult(const py::tuple &instance_outputs) {
  MSI_TIME_STAMP_START(PushPyTaskResult)
  std::vector<ResultInstance> outputs;
  ResultInstance instance;
  instance.data = PyTensor::AsInstanceData(instance_outputs);
  outputs.push_back(instance);
  Worker::GetInstance().GetWorkExecutor().GetPyTaskQueue().PyPushTaskResult(outputs);
  MSI_TIME_STAMP_END(PushPyTaskResult)
}

void PyWorker::PushPyTaskFailed(int count, const std::string &error_msg) {
  auto &task_que = Worker::GetInstance().GetWorkExecutor().GetPyTaskQueue();
  auto task_info = task_que.GetHandledTaskInfo();
  auto status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                << "Call " << task_info.tag << " Failed, method: '" << task_info.group_name
                << "', stage index(begin with 1): " << task_info.priority << ", error msg: " << error_msg;
  std::vector<ResultInstance> results;
  for (int i = 0; i < count; i++) {
    ResultInstance result_instance;
    result_instance.error_msg = status;
    results.push_back(result_instance);
  }
  task_que.PyPushTaskResult(results);
}

void PyWorker::PushPyTaskSystemFailed(const std::string &error_msg) {
  auto task_info = Worker::GetInstance().GetWorkExecutor().GetPyTaskQueue().GetHandledTaskInfo();
  auto status = INFER_STATUS_LOG_ERROR(SYSTEM_ERROR)
                << "Call " << task_info.tag << " Failed, method: '" << task_info.group_name
                << "', stage index(begin with 1): " << task_info.priority << ", error msg: " << error_msg;
  Worker::GetInstance().ClearOnSystemFailed(status);
}

void PyWorker::WaitAndClear() {
  {
    py::gil_scoped_release release;
    ExitSignalHandle::Instance().WorkerWait();
  }
  Worker::GetInstance().Clear();
}

void PyWorker::StopAndClear() {
  ExitSignalHandle::Instance().Stop();
  Worker::GetInstance().Clear();
}

std::string PyWorker::GetDeviceType(const std::string &target_device_type) {
  DeviceType target = kDeviceTypeNotSpecified;
  if (target_device_type == "cpu") {
    target = kDeviceTypeCpu;
  } else if (target_device_type == "gpu") {
    target = kDeviceTypeGpu;
  } else if (target_device_type == "ascend") {
    target = kDeviceTypeAscend;
  }
  auto device_type = InferenceLoader::Instance().GetSupportDeviceType(target, kUnknownType);
  if (device_type == kDeviceTypeAscend) {
    return "Ascend";
  }
  if (device_type == kDeviceTypeGpu) {
    return "Gpu";
  }
  if (device_type == kDeviceTypeCpu) {
    return "Cpu";
  }
  return "";
}

bool PyWorker::SupportReuseDevice() { return InferenceLoader::Instance().SupportReuseDevice(); }

void PyWorker::NotifyFailed(const std::string &master_address, const std::string &error_msg) {
  GrpcNotifyMaster::NotifyFailed(master_address, error_msg);
}
}  // namespace mindspore::serving
