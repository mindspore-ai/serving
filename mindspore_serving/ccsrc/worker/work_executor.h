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

#ifndef MINDSPORE_SERVING_WORKER_WORK_EXECUTOR_H
#define MINDSPORE_SERVING_WORKER_WORK_EXECUTOR_H

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <map>
#include "common/serving_common.h"
#include "common/instance.h"
#include "common/servable.h"
#include "worker/model_loader_base.h"
#include "worker/predict_thread.h"
#include "worker/task_queue.h"

namespace mindspore::serving {
using WorkCallBack = std::function<void(const std::vector<InstancePtr> &instances)>;

struct InferSession {
  std::vector<InstancePtr> instances;
  size_t reply_count = 0;
  WorkCallBack call_back = nullptr;
};

class WorkExecutor : public std::enable_shared_from_this<WorkExecutor> {
 public:
  WorkExecutor();
  ~WorkExecutor();

  Status Init(const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &model_loaders);
  Status Work(const RequestSpec &request_spec, const std::vector<InstanceData> &inputs,
              const WorkCallBack &on_process_done);
  void Stop();

  static uint64_t GetNextUserId();

  void ClearInstances(const Status &error_msg);
  uint64_t GetMaxBatchSize() const;

  PyTaskQueue &GetPyTaskQueue() { return py_task_queue_; }

 private:
  std::map<std::string, std::shared_ptr<ModelLoaderBase>> model_loaders_;

  bool init_flag_ = false;

  std::map<std::string, PredictThread> predict_thread_map_;
  PyTaskQueue py_task_queue_;
  CppTaskQueueThreadPool cpp_task_queue_pool_;

  std::map<uint64_t, InferSession> infer_session_map_;
  std::mutex infer_session_map_mutex_;

  bool ReplyError(const InstancePtr &context, const Status &error_msg);
  bool ReplyRequest(const std::vector<InstancePtr> &outputs);
  bool ReplyRequest(const InstancePtr &outputs);

  void OnReceiveStageInputs(const MethodSignature &method_def, uint64_t stage_index,
                            const std::vector<InstancePtr> &instances);

  static void CreateInputInstance(const MethodStage &stage, const InstancePtr &instance);
  static void CreateInputInstance(const MethodStage &stage, const std::vector<InstancePtr> &instances);
  static void CreateResultInstance(const InstancePtr &instance, const ResultInstance &result);

  void StageCallback(const std::vector<InstancePtr> &instances, const std::vector<ResultInstance> &outputs);
  void InitStageFunctionQueue();
  void InitPredictTaskQueue();
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_WORK_EXECUTOR_H
