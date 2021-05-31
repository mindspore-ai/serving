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

#include "master/servable_endpoint.h"

namespace mindspore::serving {

ServableEndPoint::ServableEndPoint(const ServableReprInfo &repr) : worker_repr_(repr) {
  running_version_number_ = worker_repr_.config_version_number;
}

ServableEndPoint::~ServableEndPoint() { Clear(); }

Status ServableEndPoint::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                                       PredictOnFinish on_finish) {
  auto method_name = request.servable_spec().method_name();
  auto it = model_thread_list_.find(method_name);
  if (it == model_thread_list_.end()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find model thread of method " << method_name;
  }
  auto status = it->second->DispatchAsync(request, reply, on_finish);
  return status;
}

Status ServableEndPoint::RegisterWorker(const ServableRegSpec &servable_spec, std::shared_ptr<WorkerContext> worker) {
  worker_contexts_.push_back(worker);
  if (running_version_number_ == 0) {
    running_version_number_ = servable_spec.version_number;
  }
  methods_ = servable_spec.methods;
  std::vector<std::string> method_names;
  for (auto &method : methods_) {
    // cppcheck-suppress useStlAlgorithm
    method_names.push_back(method.name);
  }
  if (model_thread_list_.empty()) {
    for (auto &method : methods_) {
      auto model_thread = std::make_shared<ModelThread>(servable_spec.servable_name, method.name,
                                                        servable_spec.version_number, servable_spec.batch_size, method);
      model_thread_list_.emplace(method.name, model_thread);
    }
  }
  for (auto &model_thread : model_thread_list_) {
    model_thread.second->AddWorker(worker->GetWorkerPid(), worker);
  }
  MSI_LOG_INFO << "Register worker success, " << worker_repr_.repr
               << ", version number: " << servable_spec.version_number << ", methods: " << method_names
               << ", worker address: " << worker->GetWorkerAddress();
  return SUCCESS;
}

Status ServableEndPoint::UnregisterWorker(const std::string &worker_address) {
  auto it = std::find_if(worker_contexts_.begin(), worker_contexts_.end(),
                         [worker_address](const std::shared_ptr<WorkerContext> &item) {
                           return item->GetWorkerAddress() == worker_address;
                         });
  if (it != worker_contexts_.end()) {
    auto worker = *it;
    MSI_LOG_INFO << "Unregister worker success, " << worker_repr_.repr
                 << ", version number: " << running_version_number_ << ", worker address: " << worker_address;
    for (auto &model_thread : model_thread_list_) {
      model_thread.second->DelWorker(worker->GetWorkerPid());
    }
    worker_contexts_.erase(it);
    return SUCCESS;
  }
  MSI_LOG_INFO << "Worker has already been unregistered, " << worker_repr_.repr
               << ", version number: " << running_version_number_ << ", worker address: " << worker_address;
  return FAILED;
}

void ServableEndPoint::Clear() {
  worker_contexts_.clear();
  model_thread_list_.clear();
}

}  // namespace mindspore::serving
