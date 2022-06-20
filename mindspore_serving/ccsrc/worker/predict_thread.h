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

#ifndef MINDSPORE_SERVING_WORKER_PREDICT_THREAD_H
#define MINDSPORE_SERVING_WORKER_PREDICT_THREAD_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "common/instance.h"
#include "worker/inference/inference.h"
#include "worker/task_queue.h"
#include "worker/model_loader_base.h"

namespace mindspore::serving {
struct PredictSubgraphInfo {
  std::vector<TensorInfo> input_infos;
};

struct PredictModelInfo {
  std::vector<PredictSubgraphInfo> sub_graph_infos;
  uint64_t batch_size = 0;
};

class PredictThread {
 public:
  PredictThread();
  ~PredictThread() noexcept;

  void PushPredictTask(const MethodStage &stage, const std::vector<InstancePtr> &inputs);
  void Start(const std::string &que_name, const std::shared_ptr<ModelLoaderBase> &model_loader,
             const ModelMeta &model_meta, const TaskCallBack &task_callback);
  void Stop();

  uint64_t GetBatchSize() const { return executor_info_.batch_size; }

 private:
  TaskQueue task_que_;
  std::vector<std::thread> predict_threads_;
  ModelMeta model_meta_;
  std::shared_ptr<ModelLoaderBase> model_loader_ = nullptr;
  PredictModelInfo executor_info_;

  static void ThreadFunc(PredictThread *queue);
  void Predict();

  void PredictHandle(const TaskInfo &task_info, const std::vector<InstancePtr> &instances);
  Status PredictInner(const TaskInfo &task_info, const std::vector<InstancePtr> &instances,
                      std::vector<ResultInstance> *instance_result);
  Status CheckPredictInput(uint64_t subgraph, const InstancePtr &instance);
  std::string AsGroupName(const std::string &model_key, uint64_t subgraph) const;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_WORKER_PREDICT_THREAD_H
