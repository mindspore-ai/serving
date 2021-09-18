/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SERVING_MASTER_WORKER_CONTEXT_H
#define MINDSPORE_SERVING_MASTER_WORKER_CONTEXT_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "proto/ms_worker.grpc.pb.h"
#include "common/serving_common.h"
#include "master/notify_worker/base_notify.h"

namespace mindspore::serving {

class ServableEndPoint;

enum WorkerStatus {
  kWorkerStatusNotAlive = 1,
  kWorkerStatusStarting,
  kWorkerStatusReady,
  kWorkerStatusNotifyExit,
  kWorkerStatusNotifyFailed,
  kWorkerStatusNotAvailable,
};

struct ServableReprInfo {
  std::string servable_name;
  uint32_t version_number = 0;
  std::string repr;
};

class MS_API WorkerContext : public std::enable_shared_from_this<WorkerContext> {
 public:
  WorkerContext() = default;
  ~WorkerContext() { Clear(); }
  bool HasErrorNotified() const { return status_ == kWorkerStatusNotifyFailed; }
  bool HasExitNotified() const { return status_ == kWorkerStatusNotifyExit; }
  std::string GetNotifiedError() const { return notified_error_; }
  bool HasReady() const { return status_ == kWorkerStatusReady; }
  bool IsInStarting() const { return status_ == kWorkerStatusStarting; }
  bool IsUnavailable() const { return status_ == kWorkerStatusNotAvailable; }
  void PrintStatus() const;
  uint64_t GetNormalHandledCount() const { return normal_handled_count; }
  uint64_t GetWorkerPid() const { return worker_pid_; }
  WorkerRegSpec GetWorkerSpec() const { return worker_spec_; }
  ServableReprInfo GetServableReprInfo() const { return servable_repr_; }
  std::string GetWorkerAddress() const { return worker_spec_.worker_address; }

  void InitServableReprInfo(const ServableReprInfo &repr) { servable_repr_ = repr; }
  // from py
  static std::shared_ptr<WorkerContext> PyInitWorkerContext(std::string servable_name, uint32_t version_number,
                                                            std::string repr, uint64_t worker_pid);
  void PyNotifyNotAlive();
  void PyNotifyStartFailed(const std::string &notified_error);
  void NotifyNotAvailable();
  void UpdateWorkerPid(uint64_t new_worker_pid);
  // from Dispatcher
  Status DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                       const PredictOnFinish &on_finish);
  // from worker
  void OnWorkerRegRequest(const WorkerRegSpec &worker_spec, std::shared_ptr<BaseNotifyWorker> notify);
  void OnReady();
  void OnExit();
  void OnStartError(const std::string &notified_error);
  void OnNotAvailable();
  // from py
  void OnNotAlive();
  void Clear();
  bool OwnDevice() const;

 private:
  std::mutex lock_;
  ServableReprInfo servable_repr_;
  uint32_t device_id_ = 0;
  uint64_t worker_pid_ = 0;

  // from worker register info
  WorkerRegSpec worker_spec_;
  std::shared_ptr<BaseNotifyWorker> notify_worker_ = nullptr;
  // from python env
  WorkerStatus status_ = kWorkerStatusNotAlive;
  std::string notified_error_;
  std::atomic_uint64_t request_count = 0;
  std::atomic_uint64_t total_normal_handled_count = 0;
  std::atomic_uint64_t total_abnormal_handled_count = 0;
  std::atomic_uint64_t normal_handled_count = 0;
  std::atomic_uint64_t abnormal_handled_count = 0;
};

}  // namespace mindspore::serving

#endif  // MINDSPORE_SERVING_MASTER_WORKER_CONTEXT_H
