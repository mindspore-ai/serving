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
#include "worker/sevable_base.h"
#include "worker/grpc/worker_server.h"

namespace mindspore {
namespace serving {

struct ServableWorkerContext {
  WorkerSpec worker_spec;
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

  Status Run(const proto::PredictRequest &request, proto::PredictReply *reply, DispatchCallback callback);
  Status RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply, const RequestSpec &request_spec,
                  const std::vector<InstanceData> &inputs, DispatchCallback callback);
  Status StartServable(std::shared_ptr<ServableBase> servable, std::shared_ptr<BaseNotifyMaster> notify_master);

  Status StartGrpcServer(const std::shared_ptr<MSWorkerServer> &grpc_server, const std::string &worker_ip,
                         int32_t port);

  void StopServable(bool notify_master = true);
  bool IsRunning();
  Status RegisterWorker();
  void Update();
  Status StartVersionController();
  Status AddWorker(const ServableWorkerContext &work);
  Status RemoveWorker(const ServableWorkerContext &work);

  PyTaskQueueGroup &GetPyTaskQueueGroup() { return py_task_queue_group_; }
  std::shared_ptr<TaskQueue> GetPyTaskQueuePreprocess() { return py_task_queue_group_.GetPreprocessTaskQueue(); }
  std::shared_ptr<TaskQueue> GetPyTaskQueuePostprocess() { return py_task_queue_group_.GetPostprocessTaskQueue(); }
  std::shared_ptr<TaskQueue> GetCppTaskQueuePreprocess() { return cpp_preprocess_.GetTaskQueue(); }
  std::shared_ptr<TaskQueue> GetCppTaskQueuePostprocess() { return cpp_postprocess_.GetTaskQueue(); }
  size_t GetBatchSize() const;

 private:
  std::vector<ServableWorkerContext> work_list_;
  PyTaskQueueGroup py_task_queue_group_;
  PreprocessThreadPool cpp_preprocess_;
  PostprocessThreadPool cpp_postprocess_;

  VersionController version_controller_;
  std::atomic_bool exit_notify_master_ = true;
  std::atomic_bool servable_started_ = false;
  std::atomic_flag clear_flag_ = ATOMIC_FLAG_INIT;
  std::shared_ptr<BaseNotifyMaster> notify_master_ = nullptr;
  std::shared_ptr<MSWorkerServer> worker_grpc_server_ = nullptr;

  std::shared_mutex worker_shared_lock_;

  ServableWorkerContext GetServableWorker(const RequestSpec &request_spec);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_H
