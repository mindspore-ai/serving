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
#include "common/exit_handle.h"
#include "worker/notfiy_master/grpc_notify.h"
#include "worker/local_servable/local_sevable.h"
#include "worker/distributed_worker/distributed_servable.h"
#include "worker/inference/inference.h"

namespace mindspore::serving {

void PyWorker::StartServable(const std::string &model_directory, const std::string &model_name, uint32_t version_number,
                             const std::string &master_address, const std::string &worker_address,
                             const std::string &dec_key, const std::string &dec_mode) {
  if (Worker::GetInstance().IsRunning()) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }
  auto servable = std::make_shared<LocalModelServable>();
  auto status = servable->StartServable(model_directory, model_name, version_number, dec_key, dec_mode);
  if (status != SUCCESS) {
    servable->Clear();
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = Worker::GetInstance().StartServable(servable, master_address, worker_address);
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
  auto servable = std::make_shared<DistributedServable>();
  status = Worker::GetInstance().StartDistributedGrpcServer(servable, distributed_address);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }

  status = servable->StartServable(servable_directory, servable_name, rank_table_json_file, version_number,
                                   wait_agents_time_in_seconds);
  if (status != SUCCESS) {
    servable->Clear();
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = Worker::GetInstance().StartServable(servable, master_address, worker_address);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

TaskItem PyWorker::GetPyTask() {
  TaskItem item;
  Worker::GetInstance().GetPyTaskQueueGroup().PopPyTask(&item);
  return item;
}

TaskItem PyWorker::GetPipelineTask() {
  TaskItem item;
  Worker::GetInstance().GetPyTaskQueueGroup().PopPipelineTask(&item);
  return item;
}

TaskItem PyWorker::TryGetPreprocessPyTask() {
  TaskItem item;
  Worker::GetInstance().GetPyTaskQueueGroup().TryPopPreprocessTask(&item);
  return item;
}

TaskItem PyWorker::TryGetPostprocessPyTask() {
  TaskItem item;
  Worker::GetInstance().GetPyTaskQueueGroup().TryPopPostprocessTask(&item);
  return item;
}

TaskItem PyWorker::TryGetPipelinePyTask() {
  TaskItem item;
  Worker::GetInstance().GetPyTaskQueueGroup().TryPopPipelineTask(&item);
  return item;
}
void PyWorker::PushPipelinePyResult(const py::tuple &output_batch) {
  MSI_TIME_STAMP_START(PushPipelinePyResult)
  std::vector<ResultInstance> outputs;
  for (auto &output : output_batch) {
    ResultInstance instance;
    instance.data = PyTensor::AsInstanceData(py::cast<py::tuple>(output));
    outputs.push_back(instance);
  }
  Worker::GetInstance().PushPyPipelineResult(outputs);
  MSI_TIME_STAMP_END(PushPipelinePyResult)
}

void PyWorker::PushPipelinePyFailed(int count) {
  std::vector<ResultInstance> results;
  Status error_msg(INVALID_INPUTS, "Pipeline Failed");
  for (int i = 0; i < count; i++) {
    ResultInstance result_instance;
    result_instance.error_msg = error_msg;
    results.push_back(result_instance);
  }
  Worker::GetInstance().PushPyPipelineResult(results);
}
void PyWorker::PushPreprocessPyResult(const py::tuple &output_batch) {
  MSI_TIME_STAMP_START(PushPreprocessPyResult)
  std::vector<ResultInstance> outputs;
  for (auto &output : output_batch) {
    ResultInstance instance;
    instance.data = PyTensor::AsInstanceData(py::cast<py::tuple>(output));
    outputs.push_back(instance);
  }
  Worker::GetInstance().PushPyPreprocessResult(outputs);
  MSI_TIME_STAMP_END(PushPreprocessPyResult)
}

void PyWorker::PushPreprocessPyFailed(int count) {
  std::vector<ResultInstance> results;
  Status error_msg(INVALID_INPUTS, "Preprocess Failed");
  for (int i = 0; i < count; i++) {
    ResultInstance result_instance;
    result_instance.error_msg = error_msg;
    results.push_back(result_instance);
  }
  Worker::GetInstance().PushPyPreprocessResult(results);
}

void PyWorker::PushPostprocessPyResult(const py::tuple &output_batch) {
  std::vector<ResultInstance> outputs;
  for (auto &output : output_batch) {
    ResultInstance instance;
    instance.data = PyTensor::AsInstanceData(py::cast<py::tuple>(output));
    outputs.push_back(instance);
  }
  Worker::GetInstance().PushPyPostprocessResult(outputs);
}

void PyWorker::PushPostprocessPyFailed(int count) {
  std::vector<ResultInstance> results;
  Status error_msg(INVALID_INPUTS, "Postprocess Failed");
  for (int i = 0; i < count; i++) {
    ResultInstance result_instance;
    result_instance.error_msg = error_msg;
    results.push_back(result_instance);
  }
  Worker::GetInstance().PushPyPostprocessResult(results);
}

void PyWorker::PushPreprocessPySystemFailed() {
  Worker::GetInstance().ClearOnSystemFailed(Status(SYSTEM_ERROR, "Preprocess Failed"));
}

void PyWorker::PushPostprocessPySystemFailed() {
  Worker::GetInstance().ClearOnSystemFailed(Status(SYSTEM_ERROR, "Postprocess Failed"));
}

void PyWorker::PushPipelinePySystemFailed() {
  Worker::GetInstance().ClearOnSystemFailed(Status(SYSTEM_ERROR, "Pipeline Failed"));
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

int PyWorker::GetBatchSize() { return Worker::GetInstance().GetBatchSize(); }

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
