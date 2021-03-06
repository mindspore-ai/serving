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

#include "master/dispacther.h"

#include <utility>
#include "common/proto_tensor.h"
#include "master/notify_worker/grpc_notify.h"
#include "master/notify_worker/local_notify.h"

namespace mindspore::serving {

Dispatcher::Dispatcher() {}

Dispatcher::~Dispatcher() { Clear(); }

DispatcherWorkerContext Dispatcher::GetWorkSession(const RequestSpec &request_spec) const {
  Status status;
  DispatcherWorkerContext context;
  auto it = servable_map_.find(request_spec.servable_name);
  if (it == servable_map_.end()) {
    return context;
  }
  if (request_spec.version_number > 0) {
    auto item = find_if(it->second.begin(), it->second.end(), [&](const DispatcherWorkerContext &v) {
      return v.worker_spec.version_number == request_spec.version_number;
    });
    if (item != it->second.end()) {
      context.worker_spec = item->worker_spec;
      context.notify_worker_ = item->notify_worker_;
    }
    return context;
  }
  uint64_t max_version_number = 0;
  for (const auto &item : it->second) {
    if (max_version_number < item.worker_spec.version_number) {
      context.worker_spec = item.worker_spec;
      context.notify_worker_ = item.notify_worker_;
      max_version_number = item.worker_spec.version_number;
    }
  }
  return context;
}

void Dispatcher::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                               PredictOnFinish on_finish) {
  MSI_EXCEPTION_IF_NULL(reply);
  Status status;
  (*reply->mutable_servable_spec()) = request.servable_spec();
  try {
    MSI_TIME_STAMP_START(Predict)
    status = DispatchAsyncInner(request, reply, on_finish);
    MSI_TIME_STAMP_END(Predict)
  } catch (const std::bad_alloc &ex) {
    MSI_LOG(ERROR) << "Serving Error: malloc memory failed";
    std::cout << "Serving Error: malloc memory failed" << std::endl;
  } catch (const std::runtime_error &ex) {
    MSI_LOG(ERROR) << "Serving Error: runtime error occurred: " << ex.what();
    std::cout << "Serving Error: runtime error occurred: " << ex.what() << std::endl;
  } catch (const std::exception &ex) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred: " << ex.what();
    std::cout << "Serving Error: exception occurred: " << ex.what() << std::endl;
  } catch (...) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred";
    std::cout << "Serving Error: exception occurred";
  }
  MSI_LOG(INFO) << "Finish call service Eval";

  if (status != SUCCESS) {
    GrpcTensorHelper::CreateReplyFromErrorMsg(status, reply);
    on_finish();
  }
}

Status Dispatcher::DispatchAsyncInner(const proto::PredictRequest &request, proto::PredictReply *reply,
                                      PredictOnFinish on_finish) {
  MSI_EXCEPTION_IF_NULL(reply);
  std::shared_lock<std::shared_mutex> lock(servable_shared_lock_);
  RequestSpec request_spec;
  GrpcTensorHelper::GetRequestSpec(request, &request_spec);
  auto worker = GetWorkSession(request_spec);
  if (!worker.notify_worker_) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", servable is not available";
  }
  bool find_method =
    std::any_of(worker.worker_spec.methods.begin(), worker.worker_spec.methods.end(),
                [&](const WorkerMethodInfo &method) { return method.name == request_spec.method_name; });
  if (!find_method) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", method is not available";
  }
  return worker.notify_worker_->DispatchAsync(request, reply, std::move(on_finish));
}

Status Dispatcher::RegisterServableCommon(const std::vector<WorkerSpec> &worker_specs, CreateNotifyWorkerFunc func) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);

  if (worker_specs.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable cannot be empty";
  }
  MSI_EXCEPTION_IF_NULL(func);

  for (auto &worker_spec : worker_specs) {
    if (worker_spec.servable_name.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name cannot be empty";
    }
    if (worker_spec.version_number <= 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name " << worker_spec.servable_name
                                            << " version number " << worker_spec.version_number << " cannot be 0";
    }
    auto it = servable_map_.find(worker_spec.servable_name);

    std::shared_ptr<BaseNotifyWorker> notify_worker = func(worker_spec);

    bool find_registered = false;
    if (it != servable_map_.end()) {
      auto item = find_if(it->second.begin(), it->second.end(), [&](const DispatcherWorkerContext &v) {
        return v.worker_spec.version_number == worker_spec.version_number &&
               v.worker_spec.worker_address == worker_spec.worker_address;
      });
      if (item != it->second.end()) {
        MSI_LOG_WARNING << "Servable " << worker_spec.servable_name << " version " << worker_spec.version_number
                        << " has been registered, old registered info will be replaced";
        item->worker_spec = worker_spec;
        item->notify_worker_ = notify_worker;
        find_registered = true;
      }
    }
    if (!find_registered) {
      DispatcherWorkerContext context;
      context.worker_spec = worker_spec;
      context.notify_worker_ = notify_worker;
      servable_map_[worker_spec.servable_name].push_back(context);
    }
  }
  return SUCCESS;
}

Status Dispatcher::UnregisterServableCommon(const std::string &worker_address) {
  if (clearing_flag) {
    return SUCCESS;
  }

  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  Status status;
  for (auto iter = servable_map_.begin(); iter != servable_map_.end();) {
    for (auto it = iter->second.begin(); it != iter->second.end();) {
      if (worker_address == it->worker_spec.worker_address) {
        it = iter->second.erase(it);
      } else {
        ++it;
      }
    }
    if (iter->second.size() == 0) {
      iter = servable_map_.erase(iter);
    } else {
      ++iter;
    }
  }
  return SUCCESS;
}

Status Dispatcher::AddServableCommon(const WorkerSpec &worker_spec, CreateNotifyWorkerFunc func) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  MSI_EXCEPTION_IF_NULL(func);

  if (worker_spec.servable_name.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "AddServable failed, servable name cannot be empty";
  }
  if (worker_spec.version_number <= 0) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "AddServable failed, servable name " << worker_spec.servable_name
                                          << " version number " << worker_spec.version_number << " cannot be 0";
  }
  Status status;
  auto it = servable_map_.find(worker_spec.servable_name);
  if (it != servable_map_.end()) {
    bool find = std::any_of(it->second.begin(), it->second.end(), [&](const DispatcherWorkerContext &item) {
      return item.worker_spec.version_number == worker_spec.version_number &&
             item.worker_spec.worker_address == worker_spec.worker_address;
    });
    if (find) {
      MSI_LOG_WARNING << "Servable " << worker_spec.servable_name << " version " << worker_spec.version_number
                      << " has been registered";
      return SUCCESS;
    }
  }
  DispatcherWorkerContext context;
  context.worker_spec = worker_spec;
  context.notify_worker_ = func(worker_spec);
  servable_map_[worker_spec.servable_name].push_back(context);
  return SUCCESS;
}

Status Dispatcher::RemoveServableCommon(const WorkerSpec &worker_spec) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  Status status;
  for (auto iter = servable_map_.begin(); iter != servable_map_.end();) {
    for (auto it = iter->second.begin(); it != iter->second.end();) {
      if (worker_spec.worker_address == it->worker_spec.worker_address &&
          it->worker_spec.servable_name == worker_spec.servable_name &&
          it->worker_spec.version_number == worker_spec.version_number) {
        it = iter->second.erase(it);
      } else {
        ++it;
      }
    }
    if (iter->second.size() == 0) {
      iter = servable_map_.erase(iter);
    } else {
      ++iter;
    }
  }
  return SUCCESS;
}

Status Dispatcher::RegisterServable(const proto::RegisterRequest &request, proto::RegisterReply * /*reply*/) {
  std::vector<WorkerSpec> worker_specs;
  GrpcTensorHelper::GetWorkerSpec(request, &worker_specs);
  auto create_notify_worker = [](const WorkerSpec &worker_spec) {
    std::shared_ptr<BaseNotifyWorker> notify_worker = std::make_shared<GrpcNotifyWorker>(worker_spec.worker_address);
    return notify_worker;
  };
  return RegisterServableCommon(worker_specs, create_notify_worker);
}

Status Dispatcher::UnregisterServable(const proto::ExitRequest &request, proto::ExitReply * /*reply*/) {
  return UnregisterServableCommon(request.address());
}

Status Dispatcher::AddServable(const proto::AddWorkerRequest &request, proto::AddWorkerReply * /*reply*/) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  WorkerSpec worker_spec;
  GrpcTensorHelper::GetWorkerSpec(request, &worker_spec);

  auto create_notify_worker = [](const WorkerSpec &worker_spec) {
    std::shared_ptr<BaseNotifyWorker> notify_worker = std::make_shared<GrpcNotifyWorker>(worker_spec.worker_address);
    return notify_worker;
  };
  return AddServableCommon(worker_spec, create_notify_worker);
}

Status Dispatcher::RemoveServable(const proto::RemoveWorkerRequest &request, proto::RemoveWorkerReply * /*reply*/) {
  WorkerSpec worker_spec;
  GrpcTensorHelper::GetWorkerSpec(request, &worker_spec);
  return RemoveServableCommon(worker_spec);
}

void Dispatcher::Clear() {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  clearing_flag = true;

  for (auto iter = servable_map_.begin(); iter != servable_map_.end(); ++iter) {
    for (auto it = iter->second.begin(); it != iter->second.end(); ++it) {
      if (it->notify_worker_) {
        it->notify_worker_->Exit();
      }
    }
  }
  servable_map_.clear();
}

Status Dispatcher::RegisterLocalServable(const std::vector<WorkerSpec> &worker_specs) {
  auto create_notify_worker = [](const WorkerSpec &worker_spec) {
    std::shared_ptr<BaseNotifyWorker> notify_worker = std::make_shared<LocalNotifyWorker>();
    return notify_worker;
  };
  return RegisterServableCommon(worker_specs, create_notify_worker);
}

Status Dispatcher::UnregisterLocalServable() { return UnregisterServableCommon(""); }

Status Dispatcher::AddLocalServable(const WorkerSpec &worker_spec) {
  auto create_notify_worker = [](const WorkerSpec &worker_spec) {
    std::shared_ptr<BaseNotifyWorker> notify_worker = std::make_shared<LocalNotifyWorker>();
    return notify_worker;
  };
  return AddServableCommon(worker_spec, create_notify_worker);
}

Status Dispatcher::RemoveLocalServable(const WorkerSpec &worker_spec) { return RemoveServableCommon(worker_spec); }

}  // namespace mindspore::serving
