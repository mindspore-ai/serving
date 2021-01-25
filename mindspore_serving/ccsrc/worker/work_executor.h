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
#include <unordered_map>
#include <memory>
#include <string>
#include <future>
#include <set>

#include "common/thread_pool.h"
#include "common/serving_common.h"
#include "common/instance.h"
#include "common/servable.h"
#include "worker/sevable_base.h"
#include "worker/predict_thread.h"
#include "worker/task_queue.h"

namespace mindspore {
namespace serving {

using WorkCallBack = std::function<void(const Instance &output, const Status &error_msg)>;

class WorkExecutor {
 public:
  WorkExecutor(std::shared_ptr<TaskQueue> py_preprocess_task_queue,
               std::shared_ptr<TaskQueue> py_postprocess_task_queue,
               std::shared_ptr<TaskQueue> cpp_preprocess_task_queue,
               std::shared_ptr<TaskQueue> cpp_postprocess_task_queue);
  ~WorkExecutor();

  Status Init(const ServableSignature &servable_declare, const std::shared_ptr<ServableBase> &servable);
  std::vector<std::future<void>> Work(const RequestSpec &request_spec, const std::vector<InstanceData> &inputs,
                                      WorkCallBack on_process_done);

  static uint64_t GetNextUserId();
  uint32_t GetWorkerId() const;

 private:
  std::set<std::string> python_preprocess_names_;
  std::set<std::string> python_postprocess_names_;
  PredictThread predict_thread_;

  ServableSignature servable_declare_;
  std::shared_ptr<ServableBase> servable_;
  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfo> output_infos_;
  uint32_t model_batch_size_ = 0;
  uint64_t worker_id_ = 0;
  bool init_flag_ = false;

  std::shared_ptr<TaskQueue> py_preprocess_task_queue_;
  std::shared_ptr<TaskQueue> py_postprocess_task_queue_;
  std::shared_ptr<TaskQueue> cpp_preprocess_task_queue_;
  std::shared_ptr<TaskQueue> cpp_postprocess_task_queue_;
  std::vector<TensorBasePtr> inference_inputs_;

  Status CheckSevableSignature();

  bool ReplyError(const std::vector<Instance> &context, const Status &error_msg);
  bool ReplyError(const Instance &context, const Status &error_msg);
  bool ReplyRequest(const std::vector<Instance> &outputs);
  bool ReplyRequest(const Instance &outputs);

  void OnRecievePreprocessInputs(const std::vector<Instance> &inputs);  //  callback
  void OnRecievePredictInputs(const std::vector<Instance> &inputs);     //  callback
  void OnRecievePostprocessInputs(const Instance &inputs);              //  callback

  void PredictHandle(const std::vector<Instance> &inputs);
  Status PrePredict(const std::vector<Instance> &inputs);
  Status PostPredict(const std::vector<Instance> &inputs, const std::vector<TensorBasePtr> &predict_result,
                     std::vector<Instance> *outputs);
  Status Predict(const std::vector<Instance> &inputs, std::vector<Instance> *outputs);
  Status CheckPredictInput(const Instance &instance);
  bool IsNoBatchDimInput(int input_index) const;

  Instance CreateInputInstance(const Instance &instance, PredictPhaseTag phase);
  std::vector<Instance> CreateInputInstance(const std::vector<Instance> &instance, PredictPhaseTag phase);
  Instance CreateResultInstance(const Instance &input, const ResultInstance &result, PredictPhaseTag phase);
  std::vector<Instance> CreateResultInstance(const std::vector<Instance> &inputs,
                                             const std::vector<ResultInstance> &results, PredictPhaseTag phase);
  void InitPrePostprocess();
  void InitInputTensors();
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORK_EXECUTOR_H
