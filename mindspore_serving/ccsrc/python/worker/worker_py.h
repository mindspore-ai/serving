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

#ifndef MINDSPORE_SERVING_WORKER_PY_H
#define MINDSPORE_SERVING_WORKER_PY_H

#include <string>
#include <unordered_map>
#include "common/serving_common.h"
#include "worker/worker.h"
#include "worker/task_queue.h"
#include "python/tensor_py.h"

namespace mindspore::serving {

class MS_API PyWorker {
 public:
  static void StartServable(const std::string &model_directory, const std::string &model_name, uint32_t version_number,
                            const std::string &master_address, const std::string &worker_address);

  static void StartDistributedServable(const std::string &servable_directory, const std::string &servable_name,
                                       const std::string &rank_table_json_file, uint32_t version_number,
                                       const std::string &distributed_address, const std::string &master_address,
                                       const std::string &worker_address, uint32_t wait_agents_time_in_seconds);

  static int GetBatchSize();
  static void WaitAndClear();
  static void StopAndClear();
  static TaskItem GetPyTask();
  static TaskItem GetPipelineTask();
  static TaskItem TryGetPreprocessPyTask();
  static TaskItem TryGetPostprocessPyTask();
  static TaskItem TryGetPipelinePyTask();
  static void PushPipelinePyResult(const py::tuple &output_batch);
  static void PushPipelinePyFailed(int count);
  static void PushPreprocessPyResult(const py::tuple &output_batch);
  static void PushPreprocessPyFailed(int count);

  static void PushPostprocessPyResult(const py::tuple &output_batch);
  static void PushPostprocessPyFailed(int count);
  static void PushPreprocessPySystemFailed();
  static void PushPostprocessPySystemFailed();
  static void PushPipelinePySystemFailed();
  static std::string GetDeviceType();
  // for grpc notify failed of worker
  static void NotifyFailed(const std::string &master_address, const std::string &error_msg);

 private:
  static void OnEndStartServable(const std::string &servable_directory, const std::string &servable_name,
                                 uint32_t spec_version_number, uint32_t started_version_number);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_PY_H
