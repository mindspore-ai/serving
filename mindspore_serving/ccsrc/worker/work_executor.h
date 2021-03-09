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
#include <mutex>
#include <map>
#include "common/thread_pool.h"
#include "common/serving_common.h"
#include "common/instance.h"
#include "common/servable.h"
#include "worker/sevable_base.h"
#include "worker/predict_thread.h"
#include "worker/task_queue.h"

namespace mindspore {
namespace serving {

using WorkCallBack = std::function<void(const std::vector<InstancePtr> &instances)>;

struct InferSession {
  std::vector<InstancePtr> instances;
  size_t reply_count = 0;
  WorkCallBack call_back = nullptr;
};

class WorkExecutor : public std::enable_shared_from_this<WorkExecutor> {
 public:
  WorkExecutor(std::shared_ptr<TaskQueue> py_preprocess, std::shared_ptr<TaskQueue> py_postprocess,
               std::shared_ptr<TaskQueue> cpp_preprocess, std::shared_ptr<TaskQueue> cpp_postprocess);
  ~WorkExecutor();

  Status Init(const ServableSignature &servable_declare, const std::shared_ptr<ServableBase> &servable);
  Status Work(const RequestSpec &request_spec, const std::vector<InstanceData> &inputs, WorkCallBack on_process_done);

  static uint64_t GetNextUserId();
  uint32_t GetWorkerId() const;

 private:
  std::set<std::string> python_preprocess_names_;
  std::set<std::string> python_postprocess_names_;
  PredictThread predict_thread_;

  ServableSignature servable_declare_;
  std::shared_ptr<ServableBase> servable_;
  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfoWithBatch> output_infos_;
  uint32_t model_batch_size_ = 0;
  uint64_t worker_id_ = 0;
  bool init_flag_ = false;

  std::shared_ptr<TaskQueue> py_preprocess_task_queue_;
  std::shared_ptr<TaskQueue> py_postprocess_task_queue_;
  std::shared_ptr<TaskQueue> cpp_preprocess_task_queue_;
  std::shared_ptr<TaskQueue> cpp_postprocess_task_queue_;
  std::vector<TensorBasePtr> inference_inputs_;

  std::map<uint64_t, InferSession> infer_session_map_;
  std::mutex infer_session_map_mutex_;

  void ClearInstances();

  Status CheckServableSignature();

  bool ReplyError(const std::vector<InstancePtr> &context, const Status &error_msg);
  bool ReplyError(const InstancePtr &context, const Status &error_msg);
  bool ReplyRequest(const std::vector<InstancePtr> &outputs);
  bool ReplyRequest(const InstancePtr &outputs);

  void OnReceivePreprocessInputs(const std::vector<InstancePtr> &instances);   //  callback
  void OnReceivePredictInputs(const std::vector<InstancePtr> &instances);      //  callback
  void OnReceivePostprocessInputs(const std::vector<InstancePtr> &instances);  //  callback

  void PredictHandle(const std::vector<InstancePtr> &instances);
  Status PrePredict(const std::vector<InstancePtr> &instances);
  Status PostPredict(const std::vector<InstancePtr> &instances, const std::vector<TensorBasePtr> &predict_result);
  Status Predict(const std::vector<InstancePtr> &instances);
  Status CheckPredictInput(const InstancePtr &instance);
  bool IsNoBatchDimInput(int input_index) const;

  void CreateInputInstance(const InstancePtr &instance, PredictPhaseTag phase);
  void CreateInputInstance(const std::vector<InstancePtr> &instances, PredictPhaseTag phase);
  void CreateResultInstance(const InstancePtr &instance, const ResultInstance &result, PredictPhaseTag phase);
  void CreateResultInstance(std::vector<InstancePtr> instances, const std::vector<ResultInstance> &results,
                            PredictPhaseTag phase);
  void InitPrePostprocess();
  void InitInputTensors();
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORK_EXECUTOR_H
