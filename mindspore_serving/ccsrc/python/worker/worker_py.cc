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
  for (auto &model_meta : signature.model_metas) {
    auto &model_key = model_meta.common_meta.model_key;
    auto local_models_loader = std::make_shared<LocalModelLoader>();
    status =
      local_models_loader->LoadModel(servable_directory, servable_name, version_number, model_meta, dec_key, dec_mode);
    if (status != SUCCESS) {
      local_models_loader->Clear();
      MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
    }
    status = local_models_loader->AfterLoadModel();
    if (status != SUCCESS) {
      local_models_loader->Clear();
      MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
    }
    models_loader[model_key] = local_models_loader;
  }
  status = Worker::GetInstance().StartServable(servable_directory, servable_name, version_number, models_loader,
                                               master_address, worker_address, true);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
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
                                  uint32_t version_number, const std::string &master_address,
                                  const std::string &worker_address) {
  if (Worker::GetInstance().IsRunning()) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }
  const auto &signature = ServableRegister::Instance().GetServableSignature();
  if (signature.servable_name != servable_name) {
    MSI_LOG_EXCEPTION << "Servable '" << servable_name << "' has not been registered";
  }
  std::map<std::string, std::shared_ptr<ModelLoaderBase>> model_loaders;
  Status status;
  if (!signature.model_metas.empty()) {
    status = RemoteCallModel::InitRemote(servable_name, version_number, master_address, &model_loaders);
    if (status != SUCCESS) {
      MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
    }
  }
  status = Worker::GetInstance().StartServable(servable_directory, servable_name, version_number, model_loaders,
                                               master_address, worker_address, false);
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

std::string PyWorker::GetDeviceType() {
  auto device_type = InferenceLoader::Instance().GetSupportDeviceType(kDeviceTypeNotSpecified, kUnknownType);
  if (device_type == kDeviceTypeAscendMS) {
    return "AscendMS";
  }
  if (device_type == kDeviceTypeAscendCL) {
    return "AscendCL";
  }
  if (device_type == kDeviceTypeGpu) {
    return "Gpu";
  }
  if (device_type == kDeviceTypeCpu) {
    return "Cpu";
  }
  return std::string();
}

void PyWorker::NotifyFailed(const std::string &master_address, const std::string &error_msg) {
  GrpcNotifyMaster::NotifyFailed(master_address, error_msg);
}

}  // namespace mindspore::serving
