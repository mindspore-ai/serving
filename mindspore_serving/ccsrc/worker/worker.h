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
#include "worker/notfiy_master/grpc_notify.h"
#include "common/grpc_server.h"
#include "worker/task_queue.h"
#include "common/grpc_async_server.h"
#include "worker/sevable_base.h"
#include "worker/grpc/worker_server.h"
#include "worker/distributed_worker/distributed_process/distributed_server.h"

namespace mindspore {
namespace serving {

struct ServableWorkerContext {
  ServableRegSpec servable_spec;
  ServableSignature servable_signature;
  std::shared_ptr<WorkExecutor> worker_service = nullptr;
  std::shared_ptr<ServableBase> servable = nullptr;
};

class MS_API Worker {
 public:
  Worker();
  ~Worker();

  static Worker &GetInstance();
  void Clear();
  Status Run(const RequestSpec &request_spec, const std::vector<InstanceData> instances_data,
             std::vector<InstancePtr> *out);
  Status RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish);
  Status StartServable(const std::shared_ptr<ServableBase> &servable, const std::string &master_address,
                       const std::string &worker_address);

  Status StartGrpcServer(const std::string &server_address);
  Status StartDistributedGrpcServer(std::shared_ptr<DistributedServable> servable, const std::string &server_address);

  void StopServable(bool notify_master = true);
  bool IsRunning();
  Status RegisterWorker(const std::string &master_address, const std::string &worker_address);

  PyTaskQueueGroup &GetPyTaskQueueGroup() { return py_task_queue_group_; }
  size_t GetBatchSize() const;
  void PushPyPreprocessResult(std::vector<ResultInstance> outputs);
  void PushPyPostprocessResult(std::vector<ResultInstance> outputs);
  void ClearOnSystemFailed(const Status &error_msg);
  void PushPyPipelineResult(std::vector<ResultInstance> outputs);
  void InitPipeline(const std::string &servable_name, uint64_t version_number);
  Status RunPipeline(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish);
  Status CreatePipelineInstanceFromRequest(const proto::PredictRequest &request, RequestSpec *request_spec,
                                           std::vector<InstanceData> *results);

 private:
  ServableWorkerContext servable_context_;
  ServableRegSpec pipeline_spec_;
  PyTaskQueueGroup py_task_queue_group_;
  PreprocessThreadPool cpp_preprocess_;
  PostprocessThreadPool cpp_postprocess_;

  std::atomic_bool exit_notify_master_ = true;
  std::atomic_bool servable_started_ = false;
  std::atomic_flag clear_flag_ = ATOMIC_FLAG_INIT;
  std::shared_ptr<GrpcNotifyMaster> notify_master_ = nullptr;
  std::shared_ptr<WorkerGrpcServer> worker_grpc_server_ = nullptr;
  std::shared_ptr<DistributedWorkerGrpcServer> distributed_grpc_server_ = nullptr;

  std::shared_mutex worker_shared_lock_;

  Status StartServableInner(const std::shared_ptr<ServableBase> &servable);

  Status RunAsyncInner(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish);
  bool CheckServableRequest(const RequestSpec &request_spec);
  std::shared_ptr<TaskQueue> GetPyTaskQueuePreprocess() { return py_task_queue_group_.GetPreprocessTaskQueue(); }
  std::shared_ptr<TaskQueue> GetPyTaskQueuePostprocess() { return py_task_queue_group_.GetPostprocessTaskQueue(); }
  std::shared_ptr<TaskQueue> GetCppTaskQueuePreprocess() { return cpp_preprocess_.GetTaskQueue(); }
  std::shared_ptr<TaskQueue> GetCppTaskQueuePostprocess() { return cpp_postprocess_.GetTaskQueue(); }
  std::shared_ptr<TaskQueue> GetPyTaskQueuePipeline() { return py_task_queue_group_.GetPipelineTaskQueue(); }
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_H
