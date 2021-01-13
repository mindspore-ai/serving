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
#include "worker/notfiy_master/local_notify.h"

namespace mindspore::serving {

void PyWorker::StartServable(const std::string &model_directory, const std::string &model_name, uint32_t version_number,
                             const std::string &master_ip, uint32_t master_port, const std::string &host_ip,
                             uint32_t host_port) {
  auto notify_master = std::make_shared<GrpcNotfiyMaster>(master_ip, master_port, host_ip, host_port);
  auto status = Worker::GetInstance().StartServable(model_directory, model_name, version_number, notify_master);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = Worker::GetInstance().StartGrpcServer(host_ip, host_port);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = Worker::GetInstance().StartVersionController();
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

void PyWorker::StartServableInMaster(const std::string &model_directory, const std::string &model_name,
                                     uint32_t version_number) {
  auto notify_master = std::make_shared<LocalNotifyMaster>();
  auto status = Worker::GetInstance().StartServable(model_directory, model_name, version_number, notify_master);
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
  status = Worker::GetInstance().StartVersionController();
  if (status != SUCCESS) {
    MSI_LOG_EXCEPTION << "Raise failed: " << status.StatusMessage();
  }
}

TaskItem PyWorker::GetPyTask() {
  TaskItem item;
  Worker::GetInstance().GetPyTaskQueueGroup().PopPyTask(&item);
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

void PyWorker::PushPreprocessPyResult(const py::tuple &output_batch) {
  MSI_TIME_STAMP_START(PushPreprocessPyResult)
  std::vector<ResultInstance> outputs;
  for (auto &output : output_batch) {
    ResultInstance instance;
    instance.data = PyTensor::AsInstanceData(py::cast<py::tuple>(output));
    outputs.push_back(instance);
  }
  Worker::GetInstance().GetPyTaskQueuePreprocess()->PushTaskPyResult(outputs);
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
  Worker::GetInstance().GetPyTaskQueuePreprocess()->PushTaskPyResult(results);
}

void PyWorker::PushPostprocessPyResult(const py::tuple &output_batch) {
  std::vector<ResultInstance> outputs;
  for (auto &output : output_batch) {
    ResultInstance instance;
    instance.data = PyTensor::AsInstanceData(py::cast<py::tuple>(output));
    outputs.push_back(instance);
  }
  Worker::GetInstance().GetPyTaskQueuePostprocess()->PushTaskPyResult(outputs);
}

void PyWorker::PushPostprocessPyFailed(int count) {
  std::vector<ResultInstance> results;
  Status error_msg(INVALID_INPUTS, "Postprocess Failed");
  for (int i = 0; i < count; i++) {
    ResultInstance result_instance;
    result_instance.error_msg = error_msg;
    results.push_back(result_instance);
  }
  Worker::GetInstance().GetPyTaskQueuePostprocess()->PushTaskPyResult(results);
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

}  // namespace mindspore::serving
