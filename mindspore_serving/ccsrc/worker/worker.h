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
#include "worker/model_loader_base.h"
#include "worker/grpc/worker_server.h"
#include "worker/distributed_worker/distributed_process/distributed_server.h"

namespace mindspore {
namespace serving {

class MS_API Worker {
 public:
  Worker();
  ~Worker();

  static Worker &GetInstance();
  void Clear();
  Status Run(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
             std::vector<InstancePtr> *out);
  Status RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply, const PredictOnFinish &on_finish);

  Status RunAsync(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                  const WorkCallBack &on_process_done);
  Status StartServable(const std::string &servable_directory, const std::string &servable_name, uint32_t version_number,
                       const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models,
                       const std::string &master_address, const std::string &worker_address, bool own_device);

  Status StartGrpcServer(const std::string &server_address);
  Status StartDistributedGrpcServer(std::shared_ptr<DistributedModelLoader> servable,
                                    const std::string &server_address);

  void StopServable(bool notify_master = true);
  bool IsRunning();
  Status RegisterWorker(const std::string &master_address, const std::string &worker_address);

  WorkExecutor &GetWorkExecutor() { return worker_executor_; }
  void ClearOnSystemFailed(const Status &error_msg);
  std::shared_ptr<GrpcNotifyMaster> GetGrpcNotifyMaster() { return notify_master_; }

 private:
  WorkExecutor worker_executor_;

  ServableRegSpec servable_spec_;

  std::atomic_bool exit_notify_master_ = true;
  std::atomic_bool servable_started_ = false;
  std::atomic_flag clear_flag_ = ATOMIC_FLAG_INIT;
  std::shared_ptr<GrpcNotifyMaster> notify_master_ = nullptr;
  std::shared_ptr<WorkerGrpcServer> worker_grpc_server_ = nullptr;
  std::shared_ptr<DistributedWorkerGrpcServer> distributed_grpc_server_ = nullptr;

  std::shared_mutex worker_shared_lock_;

  Status StartServableInner(const std::string &servable_name, uint32_t version_number,
                            const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models, bool own_device);

  Status RunAsyncInner(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                       const WorkCallBack &on_process_done);
  bool CheckServableRequest(const RequestSpec &request_spec);
};

}  // namespace serving
}  // namespace mindspore

#endif  // MINDSPORE_SERVING_WORKER_WORKER_H
