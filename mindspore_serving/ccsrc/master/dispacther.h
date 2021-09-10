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

#ifndef MINDSPORE_SERVING_MASTER_DISPACTHER_H
#define MINDSPORE_SERVING_MASTER_DISPACTHER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include "proto/ms_worker.grpc.pb.h"
#include "common/serving_common.h"
#include "common/instance.h"
#include "common/servable.h"
#include "master/notify_worker/base_notify.h"
#include "common/grpc_client.h"
#include "master/worker_context.h"
#include "master/servable_endpoint.h"

namespace mindspore::serving {

class MS_API Dispatcher {
 public:
  Dispatcher();
  ~Dispatcher();
  void DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish);

  Status RegisterServable(const proto::RegisterRequest &request, proto::RegisterReply *reply);
  Status NotifyWorkerExit(const proto::ExitRequest &request, proto::ExitReply *reply);
  Status NotifyWorkerFailed(const proto::NotifyFailedRequest *request, proto::NotifyFailedReply *reply);
  Status NotifyWorkerNotAlive(WorkerContext *worker_context);
  Status NotifyWorkerNotAvailable(WorkerContext *worker_context);
  void GetModelInfo(const proto::GetModelInfoRequest *request, proto::GetModelInfoReply *reply);
  void Clear();

  std::shared_ptr<WorkerContext> InitWorkerContext(const ServableReprInfo &repr, uint64_t worker_pid);
  bool OnlyModelStage(const std::string &servable_name);

 private:
  std::vector<std::shared_ptr<ServableEndPoint>> servable_list_;
  std::vector<std::shared_ptr<WorkerContext>> worker_list_;

  std::shared_mutex servable_shared_lock_;
  std::atomic_uint32_t enqueued_requests_ = 0;

  Status JudgeInferNum();
  std::shared_ptr<ServableEndPoint> GetWorkerEndpoint(const RequestSpec &request_spec) const;

  using CreateNotifyWorkerFunc = std::function<std::shared_ptr<BaseNotifyWorker>(const WorkerRegSpec &worker_spec)>;

  Status RegisterServableCommon(const WorkerRegSpec &worker_spec, CreateNotifyWorkerFunc func);
  Status UnregisterServableCommon(const std::string &worker_address);
  Status DispatchAsyncInner(const proto::PredictRequest &request, proto::PredictReply *reply,
                            PredictOnFinish on_finish);
  Status RegisterWorkerContext(std::shared_ptr<WorkerContext> worker_context);
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_MASTER_DISPACTHER_H
