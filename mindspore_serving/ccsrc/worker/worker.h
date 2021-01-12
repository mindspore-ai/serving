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

#ifndef MINDSPORE_SERVING_WORKER_WORKER_H
#define MINDSPORE_SERVING_WORKER_WORKER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include <shared_mutex>
#include <map>
#include "worker/work_executor.h"
#include "common/serving_common.h"
#include "proto/ms_service.pb.h"
#include "worker/notfiy_master/base_notify.h"
#include "common/grpc_server.h"
#include "worker/task_queue.h"
#include "worker/version_control/version_controller.h"
#include "common/grpc_async_server.h"

namespace mindspore {
namespace serving {

class AsyncResult {
 public:
  explicit AsyncResult(size_t size);

  bool HasNext();
  Status GetNext(Instance *instance_result);

 private:
  std::vector<std::future<void>> future_list_;
  std::vector<Instance> result_;
  size_t next_index_;
  bool time_out_last_ = false;

  friend class Worker;
};

struct ServableWorkerContext {
  LoadServableSpec servable_spec;
  ServableSignature servable_signature;
  std::shared_ptr<WorkExecutor> worker_service = nullptr;
  uint32_t model_id = 0;
  std::string model_file_name;
};

class MS_API Worker {
 public:
  Worker();
  ~Worker();

  static Worker &GetInstance();
  void Clear();

  Status Run(const proto::PredictRequest &request, proto::PredictReply *reply);
  Status Run(const RequestSpec &request_spec, const std::vector<InstanceData> &inputs, std::vector<Instance> *outputs);
  std::pair<Status, std::shared_ptr<AsyncResult>> RunAsync(const RequestSpec &request_spec,
                                                           const std::vector<InstanceData> &inputs);

  Status InitEnv(ModelType model_type, const std::map<std::string, std::string> &other_options);
  Status FinalizeEnv();

  Status StartServable(const std::string &servable_directory, const std::string &servable_name, uint32_t version_number,
                       std::shared_ptr<BaseNotifyMaster> notify_master);
  void StopServable(bool notify_master = true);
  bool HasCleared();
  Status RegisterWorker();
  Status StartGrpcServer(const std::string &ip, uint32_t grpc_port);
  Status LoadModel(LoadServableSpec *servable_spec, uint64_t version, ServableWorkerContext *work);
  void Update();
  Status StartVersionController();
  Status AddWorker(const ServableWorkerContext &work);
  Status RemoveWorker(const ServableWorkerContext &work);

  PyTaskQueueGroup &GetPyTaskQueueGroup() { return py_task_queue_group_; }
  std::shared_ptr<TaskQueue> GetPyTaskQueuePreprocess() { return py_task_queue_group_.GetPreprocessTaskQueue(); }
  std::shared_ptr<TaskQueue> GetPyTaskQueuePostprocess() { return py_task_queue_group_.GetPostprocessTaskQueue(); }
  std::shared_ptr<TaskQueue> GetCppTaskQueuePreprocess() { return cpp_preprocess_.GetTaskQueue(); }
  std::shared_ptr<TaskQueue> GetCppTaskQueuePostprocess() { return cpp_postprocess_.GetTaskQueue(); }
  ssize_t GetBatchSize() const;

 private:
  static std::shared_ptr<Worker> global_worker_;

  std::vector<ServableWorkerContext> work_list_;
  std::shared_ptr<serving::InferSession> session_ = nullptr;
  std::string version_strategy_;
  PyTaskQueueGroup py_task_queue_group_;
  PreprocessThreadPool cpp_preprocess_;
  PostprocessThreadPool cpp_postprocess_;

  VersionController version_controller_;
  LoadServableSpec base_spec_;
  std::atomic_bool exit_notify_master_ = true;
  std::atomic_bool servable_started_ = false;
  std::atomic_flag clear_flag_ = ATOMIC_FLAG_INIT;
  std::shared_ptr<BaseNotifyMaster> notify_master_ = nullptr;

  std::shared_mutex worker_shared_lock_;

  ServableWorkerContext GetServableWorker(const RequestSpec &request_spec);
  Status LoadServableConfig(const LoadServableSpec &servable_spec, const std::string &version_strategy,
                            std::vector<uint64_t> *real_version_number);
  void GetVersions(const LoadServableSpec &servable_spec, std::vector<uint64_t> *real_versions);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_H
