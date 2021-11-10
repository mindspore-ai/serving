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
#include "master/master_context.h"
#include "master/notify_worker/grpc_notify.h"

namespace mindspore::serving {

Dispatcher::Dispatcher() {}

Dispatcher::~Dispatcher() { Clear(); }

std::shared_ptr<ServableEndPoint> Dispatcher::GetWorkerEndpoint(const RequestSpec &request_spec) const {
  Status status;
  if (request_spec.version_number > 0) {
    auto item = find_if(servable_list_.begin(), servable_list_.end(), [&](const std::shared_ptr<ServableEndPoint> &v) {
      return v->GetServableName() == request_spec.servable_name && v->GetVersionNumber() == request_spec.version_number;
    });
    if (item != servable_list_.end()) {
      return *item;
    }
    return nullptr;
  }
  uint64_t max_version_number = 0;
  std::shared_ptr<ServableEndPoint> endpoint = nullptr;
  for (const auto &item : servable_list_) {
    if (item->GetServableName() == request_spec.servable_name && max_version_number < item->GetVersionNumber()) {
      endpoint = item;
      max_version_number = item->GetVersionNumber();
    }
  }
  return endpoint;
}

Status Dispatcher::JudgeInferNum() {
  auto max_enqueued_requests = MasterContext::Instance()->GetMaxEnqueuedRequests();
  if (enqueued_requests_ >= max_enqueued_requests) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Serving Error: enqueued requests count exceeds the limit " << max_enqueued_requests;
  }
  return SUCCESS;
}

void Dispatcher::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                               const PredictOnFinish &on_finish) {
  MSI_EXCEPTION_IF_NULL(reply);
  (*reply->mutable_servable_spec()) = request.servable_spec();
  Status status = JudgeInferNum();
  if (status != SUCCESS) {
    GrpcTensorHelper::CreateReplyFromErrorMsg(status, reply);
    on_finish();
    return;
  }
  try {
    auto callback = [this, on_finish]() {
      on_finish();
      this->enqueued_requests_--;
    };
    enqueued_requests_++;
    status = DispatchAsyncInner(request, reply, callback);
  } catch (const std::bad_alloc &ex) {
    MSI_LOG(ERROR) << "Serving Error: malloc memory failed";
  } catch (const std::runtime_error &ex) {
    MSI_LOG(ERROR) << "Serving Error: runtime error occurred: " << ex.what();
  } catch (const std::exception &ex) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred: " << ex.what();
  } catch (...) {
    MSI_LOG(ERROR) << "Serving Error: exception occurred";
  }
  if (status != SUCCESS) {
    GrpcTensorHelper::CreateReplyFromErrorMsg(status, reply);
    on_finish();
    enqueued_requests_--;
  }
}

Status Dispatcher::DispatchAsyncInner(const proto::PredictRequest &request, proto::PredictReply *reply,
                                      const PredictOnFinish &on_finish) {
  MSI_EXCEPTION_IF_NULL(reply);
  std::shared_lock<std::shared_mutex> lock(servable_shared_lock_);
  RequestSpec request_spec;
  GrpcTensorHelper::GetRequestSpec(request, &request_spec);
  auto endpoint = GetWorkerEndpoint(request_spec);
  if (endpoint == nullptr) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", servable is not available";
  }
  auto methods = endpoint->GetMethods();
  bool find_method = std::any_of(methods.begin(), methods.end(), [&](const ServableMethodInfo &method) {
    return method.name == request_spec.method_name;
  });
  if (!find_method) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Request " << request_spec.Repr() << ", method is not available";
  }
  return endpoint->DispatchAsync(request, reply, on_finish);
}

Status Dispatcher::UnregisterServableCommon(const std::string &worker_address) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  std::shared_ptr<WorkerContext> worker_context = nullptr;
  for (auto &item : worker_list_) {
    if (item->GetWorkerAddress() == worker_address) {
      worker_context = item;
      break;
    }
  }
  if (worker_context == nullptr) {
    MSI_LOG_ERROR << "Cannot find worker context of address " << worker_address;
    return FAILED;
  }
  auto servable_spec = worker_context->GetWorkerSpec().servable_spec;
  std::shared_ptr<ServableEndPoint> endpoint = nullptr;
  for (auto &item : servable_list_) {
    if (item->GetServableName() == servable_spec.servable_name &&
        item->GetVersionNumber() == servable_spec.version_number) {
      endpoint = item;
      break;
    }
  }
  if (endpoint) {
    endpoint->UnregisterWorker(worker_address);
  }
  worker_context->OnExit();
  MSI_LOG_INFO << "Unregister worker exit success, worker pid: " << worker_context->GetWorkerPid()
               << ", worker address: " << worker_context->GetWorkerAddress();
  return SUCCESS;
}

Status Dispatcher::RegisterServable(const proto::RegisterRequest &request, proto::RegisterReply *) {
  WorkerRegSpec worker_spec;
  GrpcTensorHelper::ConvertProtoWorkerSpec(request, &worker_spec);
  auto create_notify_worker = [](const WorkerRegSpec &worker_spec) {
    std::shared_ptr<BaseNotifyWorker> notify_worker = std::make_shared<GrpcNotifyWorker>(worker_spec.worker_address);
    return notify_worker;
  };
  return RegisterServableCommon(worker_spec, create_notify_worker);
}

Status Dispatcher::NotifyWorkerExit(const proto::ExitRequest &request, proto::ExitReply *) {
  return UnregisterServableCommon(request.address());
}

Status Dispatcher::NotifyWorkerNotAlive(WorkerContext *worker_context) {
  MSI_EXCEPTION_IF_NULL(worker_context);
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  auto worker_spec = worker_context->GetWorkerSpec();
  auto &servable_spec = worker_spec.servable_spec;
  std::shared_ptr<ServableEndPoint> endpoint = nullptr;
  for (auto &item : servable_list_) {
    if (item->GetServableName() == servable_spec.servable_name &&
        item->GetVersionNumber() == servable_spec.version_number) {
      endpoint = item;
      break;
    }
  }
  if (endpoint) {
    endpoint->UnregisterWorker(worker_context->GetWorkerAddress());
  }
  worker_context->OnNotAlive();
  return SUCCESS;
}

Status Dispatcher::NotifyWorkerNotAvailable(WorkerContext *worker_context) {
  MSI_EXCEPTION_IF_NULL(worker_context);
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  auto worker_spec = worker_context->GetWorkerSpec();
  auto &servable_spec = worker_spec.servable_spec;
  std::shared_ptr<ServableEndPoint> endpoint = nullptr;
  for (auto &item : servable_list_) {
    if (item->GetServableName() == servable_spec.servable_name &&
        item->GetVersionNumber() == servable_spec.version_number) {
      endpoint = item;
      break;
    }
  }
  if (endpoint) {
    endpoint->UnregisterWorker(worker_context->GetWorkerAddress());
  }
  worker_context->OnNotAvailable();
  return SUCCESS;
}

void Dispatcher::GetModelInfo(const proto::GetModelInfoRequest *request, proto::GetModelInfoReply *reply) {
  auto &servable_name = request->servable_name();
  auto version_number = request->version_number();
  for (auto &worker : worker_list_) {
    auto worker_spec = worker->GetWorkerSpec();
    if (worker_spec.servable_spec.servable_name == servable_name &&
        worker_spec.servable_spec.version_number == version_number && worker_spec.servable_spec.own_device) {
      reply->set_servable_name(servable_name);
      reply->set_version_number(version_number);
      GrpcTensorHelper::ConvertModelInfos(worker_spec.servable_spec.models, reply->mutable_model_infos());
      return;
    }
  }
  auto status = INFER_STATUS_LOG_ERROR(FAILED)
                << "Servable '" << servable_name << "' has models declared by declare_model, but parameter 'device_ids'"
                << " of ServableStartConfig is not set in Serving startup script when the device target is not CPU";
  auto error_msg = reply->mutable_error_msg();
  error_msg->set_error_code(FAILED);
  error_msg->set_error_msg(status.StatusMessage());
}

bool Dispatcher::OnlyModelStage(const std::string &servable_name) {
  for (auto &worker : worker_list_) {
    auto worker_spec = worker->GetWorkerSpec();
    if (worker_spec.servable_spec.servable_name != servable_name) {
      continue;
    }
    for (auto &method : worker_spec.servable_spec.methods) {
      // cppcheck-suppress useStlAlgorithm
      if (!method.only_model_stage) {
        return false;
      }
    }
    return true;
  }
  return false;
}

void Dispatcher::Clear() {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);

  for (auto &endpoint : servable_list_) {
    endpoint->Clear();
  }
  for (auto &worker : worker_list_) {
    worker->Clear();
  }
  servable_list_.clear();
  worker_list_.clear();
}

Status Dispatcher::RegisterServableCommon(const WorkerRegSpec &worker_spec, CreateNotifyWorkerFunc func) {
  MSI_EXCEPTION_IF_NULL(func);
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  std::shared_ptr<WorkerContext> worker_context = nullptr;
  for (auto &item : worker_list_) {
    if (item->GetWorkerPid() == worker_spec.worker_pid) {
      worker_context = item;
      break;
    }
  }
  bool ready = true;
  if (worker_context == nullptr) {
    worker_context = std::make_shared<WorkerContext>();
    worker_context->UpdateWorkerPid(worker_spec.worker_pid);
    worker_list_.push_back(worker_context);
    ready = false;
  }
  worker_context->OnWorkerRegRequest(worker_spec, func(worker_spec));
  if (ready) {
    auto status = RegisterWorkerContext(worker_context);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Registered worker failed";
      worker_context->OnStartError("Registered worker failed");
    }
  }
  return SUCCESS;
}

Status Dispatcher::NotifyWorkerFailed(const proto::NotifyFailedRequest *request, proto::NotifyFailedReply *reply) {
  auto worker_pid = request->worker_pid();
  auto error_msg = request->error_msg();
  MSI_LOG_ERROR << "Worker notify failed, worker pid: " << worker_pid << ", error reported: <" << error_msg << ">";
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  std::shared_ptr<WorkerContext> worker_context = nullptr;
  for (auto &item : worker_list_) {
    if (item->GetWorkerPid() == worker_pid) {
      worker_context = item;
      break;
    }
  }
  if (worker_context == nullptr) {
    worker_context = std::make_shared<WorkerContext>();
    worker_context->UpdateWorkerPid(worker_pid);
    worker_list_.push_back(worker_context);
  }
  worker_context->OnStartError(error_msg);
  return SUCCESS;
}

std::shared_ptr<WorkerContext> Dispatcher::InitWorkerContext(const ServableReprInfo &repr, uint64_t worker_pid) {
  std::unique_lock<std::shared_mutex> lock(servable_shared_lock_);
  std::shared_ptr<WorkerContext> worker_context = nullptr;
  for (auto &item : worker_list_) {
    if (item->GetWorkerPid() == worker_pid) {
      worker_context = item;
      break;
    }
  }
  bool ready = true;
  if (worker_context == nullptr) {
    worker_context = std::make_shared<WorkerContext>();
    worker_context->UpdateWorkerPid(worker_pid);
    worker_list_.push_back(worker_context);
    ready = false;
  }
  worker_context->InitServableReprInfo(repr);
  if (ready) {
    auto status = RegisterWorkerContext(worker_context);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Registered worker failed";
      worker_context->OnStartError("Registered worker failed");
    }
  }
  return worker_context;
}

Status Dispatcher::RegisterWorkerContext(std::shared_ptr<WorkerContext> worker_context) {
  auto worker_spec = worker_context->GetWorkerSpec();
  auto &servable_spec = worker_spec.servable_spec;
  if (servable_spec.servable_name.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name cannot be empty";
  }
  if (servable_spec.version_number <= 0) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Register failed, servable name " << servable_spec.servable_name
                                          << " version number " << servable_spec.version_number << " cannot be 0";
  }
  std::shared_ptr<ServableEndPoint> endpoint = nullptr;
  for (auto &item : servable_list_) {
    if (item->GetServableName() == servable_spec.servable_name &&
        item->GetVersionNumber() == servable_spec.version_number) {
      endpoint = item;
      break;
    }
  }
  if (!endpoint) {
    endpoint = std::make_shared<ServableEndPoint>(worker_context->GetServableReprInfo());
    servable_list_.push_back(endpoint);
  }
  endpoint->RegisterWorker(servable_spec, worker_context);
  worker_context->OnReady();
  return SUCCESS;
}

}  // namespace mindspore::serving
