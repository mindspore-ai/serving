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
#include "worker/worker.h"
#include "common/proto_tensor.h"

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
      context.stub_ = item->stub_;
      context.worker_running_in_master = item->worker_running_in_master;
    }
    return context;
  }
  uint64_t max_version_number = 0;
  for (const auto &item : it->second) {
    if (max_version_number < item.worker_spec.version_number) {
      context.worker_spec = item.worker_spec;
      context.stub_ = item.stub_;
      context.worker_running_in_master = item.worker_running_in_master;
      max_version_number = item.worker_spec.version_number;
    }
  }
  return context;
}

Status Dispatcher::Dispatch(const proto::PredictRequest &request, proto::PredictReply *reply) {
  MSI_EXCEPTION_IF_NULL(reply);
  std::shared_lock<std::shared_mutex> lock(servable_shared_lock_);
  RequestSpec request_spec;
  GrpcTensorHelper::GetRequestSpec(request, &request_spec);
  auto worker = GetWorkSession(request_spec);
  if (!worker.stub_ && !worker.worker_running_in_master) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", servable is not available";
  }
  bool find_method =
    std::any_of(worker.worker_spec.methods.begin(), worker.worker_spec.methods.end(),
                [&](const WorkerMethodInfo &method) { return method.name == request_spec.method_name; });
  if (!find_method) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", method is not available";
  }
  /// TODO spec request version_number
  if (worker.stub_ != nullptr) {
    grpc::ClientContext context;
    auto status = worker.stub_->Predict(&context, request, reply);
    if (!status.ok()) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Predict failed, worker gRPC error: " << status.error_code() << ", " << status.error_message();
    }
  } else {
    return Worker::GetInstance().Run(request, reply);
  }
  return SUCCESS;
}

Status Dispatcher::RegisterServable(const proto::RegisterRequest &request, proto::RegisterReply * /*reply*/) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  std::vector<WorkerSpec> worker_specs;
  GrpcTensorHelper::GetWorkerSpec(request, &worker_specs);
  if (worker_specs.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable cannot be empty";
  }
  for (auto &worker_spec : worker_specs) {
    if (worker_spec.servable_name.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name cannot be empty";
    }
    if (worker_spec.version_number <= 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name " << worker_spec.servable_name
                                            << " version number " << worker_spec.version_number << " cannot be 0";
    }
    auto target_str = request.address();
    auto it = servable_map_.find(worker_spec.servable_name);

    std::shared_ptr<grpc::Channel> channel = GrpcServer::CreateChannel(target_str);

    bool find_registered = false;
    if (it != servable_map_.end()) {
      std::shared_ptr<Worker> worker;
      auto item = find_if(it->second.begin(), it->second.end(), [&](const DispatcherWorkerContext &v) {
        return v.worker_spec.version_number == worker_spec.version_number &&
               v.worker_spec.worker_address == worker_spec.worker_address;
      });
      if (item != it->second.end()) {
        MSI_LOG_WARNING << "Servable " << worker_spec.servable_name << " version " << worker_spec.version_number
                        << " has been registered, old registered info will be replaced";
        item->worker_spec = worker_spec;
        item->stub_ = proto::MSWorker::NewStub(channel);
        find_registered = true;
      }
    }
    if (!find_registered) {
      DispatcherWorkerContext context;
      context.worker_spec = worker_spec;
      context.stub_ = proto::MSWorker::NewStub(channel);
      servable_map_[worker_spec.servable_name].push_back(context);
    }
  }
  return SUCCESS;
}

Status Dispatcher::UnregisterServable(const proto::ExitRequest &request, proto::ExitReply * /*reply*/) {
  if (clearing_flag) {
    return SUCCESS;
  }
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  auto target_str = request.address();
  Status status;
  for (auto iter = servable_map_.begin(); iter != servable_map_.end();) {
    for (auto it = iter->second.begin(); it != iter->second.end();) {
      if (target_str == it->worker_spec.worker_address) {
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
Status Dispatcher::AddServable(const proto::AddWorkerRequest &request, proto::AddWorkerReply * /*reply*/) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  WorkerSpec worker_spec;
  GrpcTensorHelper::GetWorkerSpec(request, &worker_spec);
  auto target_str = request.address();
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

  std::shared_ptr<grpc::Channel> channel = GrpcServer::CreateChannel(target_str);
  context.stub_ = proto::MSWorker::NewStub(channel);
  servable_map_[worker_spec.servable_name].push_back(context);
  return SUCCESS;
}

Status Dispatcher::RemoveServable(const proto::RemoveWorkerRequest &request, proto::RemoveWorkerReply * /*reply*/) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  WorkerSpec worker_spec;
  GrpcTensorHelper::GetWorkerSpec(request, &worker_spec);
  auto target_str = request.address();
  Status status;
  for (auto iter = servable_map_.begin(); iter != servable_map_.end();) {
    for (auto it = iter->second.begin(); it != iter->second.end();) {
      if (target_str == it->worker_spec.worker_address && it->worker_spec.servable_name == worker_spec.servable_name &&
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

void Dispatcher::Clear() {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  clearing_flag = true;

  for (auto iter = servable_map_.begin(); iter != servable_map_.end(); ++iter) {
    for (auto it = iter->second.begin(); it != iter->second.end(); ++it) {
      proto::ExitRequest request;
      request.set_address(it->worker_spec.worker_address);
      proto::ExitReply reply;
      grpc::ClientContext context;
      const int32_t TIME_OUT = 1;
      std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(TIME_OUT);
      context.set_deadline(deadline);
      if (it->stub_) {
        (void)it->stub_->Exit(&context, request, &reply);
      } else {
        Worker::GetInstance().Clear();
      }
    }
  }
  servable_map_.clear();
}

Status Dispatcher::RegisterLocalServable(const std::vector<WorkerSpec> &worker_specs) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  if (worker_specs.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable cannot be empty";
  }
  for (auto &worker_spec : worker_specs) {
    if (worker_spec.servable_name.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name cannot be empty";
    }
    if (worker_spec.version_number <= 0) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name " << worker_spec.servable_name
                                            << " version number " << worker_spec.version_number << " cannot be 0";
    }
    auto it = servable_map_.find(worker_spec.servable_name);

    bool find_registered = false;
    if (it != servable_map_.end()) {
      std::shared_ptr<Worker> worker;
      auto item = find_if(it->second.begin(), it->second.end(), [&](const DispatcherWorkerContext &v) {
        return v.worker_spec.version_number == worker_spec.version_number &&
               v.worker_spec.worker_address == worker_spec.worker_address;
      });
      if (item != it->second.end()) {
        MSI_LOG_WARNING << "Servable " << worker_spec.servable_name << " version " << worker_spec.version_number
                        << " has been registered, old registered info will be replaced";
        item->worker_spec = worker_spec;
        find_registered = true;
      }
    }
    if (!find_registered) {
      DispatcherWorkerContext context;
      context.worker_spec = worker_spec;
      context.worker_running_in_master = true;
      servable_map_[worker_spec.servable_name].push_back(context);
    }
  }
  return SUCCESS;
}

Status Dispatcher::UnregisterLocalServable() {
  if (clearing_flag) {
    return SUCCESS;
  }
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  Status status;
  for (auto iter = servable_map_.begin(); iter != servable_map_.end();) {
    for (auto it = iter->second.begin(); it != iter->second.end();) {
      if (it->worker_running_in_master) {
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

Status Dispatcher::AddLocalServable(const WorkerSpec &worker_spec) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
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
  context.worker_running_in_master = true;
  servable_map_[worker_spec.servable_name].push_back(context);
  return SUCCESS;
}

Status Dispatcher::RemoveLocalServable(const WorkerSpec &worker_spec) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  Status status;
  for (auto iter = servable_map_.begin(); iter != servable_map_.end();) {
    for (auto it = iter->second.begin(); it != iter->second.end();) {
      if (it->worker_running_in_master && it->worker_spec.servable_name == worker_spec.servable_name &&
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

}  // namespace mindspore::serving
