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

#include "master/worker_context.h"
#include "master/servable_endpoint.h"
#include "master/server.h"

namespace mindspore::serving {

// from py
std::shared_ptr<WorkerContext> WorkerContext::PyInitWorkerContext(std::string servable_name, uint32_t version_number,
                                                                  std::string repr, uint64_t worker_pid) {
  ServableReprInfo servable_repr;
  servable_repr.servable_name = servable_name;
  servable_repr.version_number = version_number;
  servable_repr.repr = repr;
  return Server::Instance().GetDispatcher()->InitWorkerContext(servable_repr, worker_pid);
}

// from Dispatcher
Status WorkerContext::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                                    const PredictOnFinish &on_finish) {
  auto shared_this = shared_from_this();
  PredictOnFinish callback = [shared_this, on_finish, reply]() {
    auto &error_msg = reply->error_msg();
    auto has_error =
      std::any_of(error_msg.begin(), error_msg.end(), [](const proto::ErrorMsg &msg) { return msg.error_code() != 0; });
    if (!has_error && reply->instances_size() != 0) {
      shared_this->normal_handled_count += 1;
      shared_this->total_normal_handled_count += 1;
    } else {
      shared_this->abnormal_handled_count += 1;
      shared_this->total_abnormal_handled_count += 1;
    }
    on_finish();
  };
  std::unique_lock<std::mutex> lock(lock_);
  if (status_ != kWorkerStatusReady && !notify_worker_) {
    return INFER_STATUS_LOG_ERROR(WORKER_UNAVAILABLE) << "Worker is not ready";
  }
  request_count += 1;
  return notify_worker_->DispatchAsync(request, reply, callback);
}

// from worker
void WorkerContext::OnWorkerRegRequest(const WorkerRegSpec &worker_spec, std::shared_ptr<BaseNotifyWorker> notify) {
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_INFO << "Receive worker registered message, " << servable_repr_.repr << ", worker pid: " << worker_pid_
               << ", worker address: " << worker_spec.worker_address;
  worker_spec_ = worker_spec;
  notify_worker_ = notify;
}

void WorkerContext::OnReady() {
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_INFO << "Notify worker ready, " << servable_repr_.repr << ", worker pid: " << worker_pid_
               << ", worker address: " << worker_spec_.worker_address;
  status_ = kWorkerStatusReady;
}

void WorkerContext::OnExit() {
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_INFO << "Notify worker exit, " << servable_repr_.repr << ", worker pid: " << worker_pid_
               << ", worker address: " << worker_spec_.worker_address;
  status_ = kWorkerStatusNotifyExit;
  notify_worker_ = nullptr;
}

void WorkerContext::OnStartError(const std::string &notified_error) {
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_ERROR << "Notify worker start-up error, " << servable_repr_.repr << ", worker pid: " << worker_pid_;
  status_ = kWorkerStatusNotifyFailed;
  notify_worker_ = nullptr;
  notified_error_ = notified_error;
}

void WorkerContext::OnNotAvailable() {
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_ERROR << "Notify worker not available, " << servable_repr_.repr << ", worker pid: " << worker_pid_;
  if (status_ != kWorkerStatusNotifyExit && status_ != kWorkerStatusNotAlive) {
    status_ = kWorkerStatusNotAvailable;
  }
  notify_worker_ = nullptr;
}

void WorkerContext::OnNotAlive() {
  if (HasExitNotified() || HasErrorNotified()) {
    return;
  }
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_INFO << "Notify worker not alive, " << servable_repr_.repr << ", worker pid: " << worker_pid_
               << ", worker address: " << worker_spec_.worker_address;
  if (status_ != kWorkerStatusNotifyExit) {
    status_ = kWorkerStatusNotAlive;
  }
  notify_worker_ = nullptr;
}

// from py
void WorkerContext::PyNotifyNotAlive() { Server::Instance().GetDispatcher()->NotifyWorkerNotAlive(this); }
void WorkerContext::PyNotifyStartFailed(const std::string &notified_error) { OnStartError(notified_error); }
void WorkerContext::NotifyNotAvailable() { Server::Instance().GetDispatcher()->NotifyWorkerNotAvailable(this); }

void WorkerContext::UpdateWorkerPid(uint64_t new_worker_pid) {
  std::unique_lock<std::mutex> lock(lock_);
  MSI_LOG_INFO << "Update worker pid from " << worker_pid_ << " to " << new_worker_pid;
  if (status_ != kWorkerStatusReady) {
    status_ = kWorkerStatusStarting;
  }
  worker_pid_ = new_worker_pid;
  normal_handled_count = 0;
  abnormal_handled_count = 0;
}

void WorkerContext::Clear() {
  std::unique_lock<std::mutex> lock(lock_);
  notify_worker_ = nullptr;
  status_ = kWorkerStatusNotAlive;
}

bool WorkerContext::OwnDevice() const { return worker_spec_.servable_spec.own_device; }

void WorkerContext::PrintStatus() const {
  auto repr = servable_repr_.repr;
  switch (status_) {
    case kWorkerStatusNotAlive:
      MSI_LOG_INFO << "worker " << GetWorkerPid() << " status is kWorkerStatusNotAlive, " << repr;
      break;
    case kWorkerStatusStarting:
      MSI_LOG_INFO << "worker " << GetWorkerPid() << " status is kWorkerStatusStarting, " << repr;
      break;
    case kWorkerStatusReady:
      MSI_LOG_INFO << "worker " << GetWorkerPid() << " status is kWorkerStatusReady, " << repr;
      break;
    case kWorkerStatusNotifyExit:
      MSI_LOG_INFO << "worker " << GetWorkerPid() << " status is kWorkerStatusNotifyExit, " << repr;
      break;
    case kWorkerStatusNotifyFailed:
      MSI_LOG_INFO << "worker " << GetWorkerPid() << " status is kWorkerStatusNotifyFailed, " << repr;
      break;
    case kWorkerStatusNotAvailable:
      MSI_LOG_INFO << "worker " << GetWorkerPid() << " status is kWorkerStatusNotAvailable, " << repr;
      break;
  }
}

}  // namespace mindspore::serving
