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

#include "worker/worker.h"
#include <atomic>
#include <condition_variable>
#include <set>
#include <utility>
#include <map>
#include "pybind11/pybind11.h"
#include "common/proto_tensor.h"
#include "common/file_system_operation.h"
#include "common/exit_handle.h"
#include "worker/context.h"
#include "worker/grpc/worker_process.h"
#include "worker/task_queue.h"
#include "worker/grpc/worker_server.h"

namespace py = pybind11;

namespace mindspore {
namespace serving {

Worker &Worker::GetInstance() {
  static Worker instance;
  return instance;
}

Status Worker::RegisterWorker() {
  std::vector<WorkerSpec> worker_specs;
  for (auto &work : work_list_) {
    // cppcheck-suppress useStlAlgorithm
    worker_specs.push_back(work.worker_spec);
  }
  auto status = notify_master_->Register(worker_specs);
  return status;
}

Status Worker::StartVersionController() {
  // first disable auto updated
  return SUCCESS;
}

Status Worker::AddWorker(const ServableWorkerContext &work) { return notify_master_->AddWorker(work.worker_spec); }

Status Worker::RemoveWorker(const ServableWorkerContext &work) {
  return notify_master_->RemoveWorker(work.worker_spec);
}

Status Worker::Run(const proto::PredictRequest &request, proto::PredictReply *reply, DispatchCallback callback) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Run worker for inference failed, worker has not been started";
  }
  MSI_EXCEPTION_IF_NULL(reply);
  std::vector<InstanceData> inputs;
  RequestSpec request_spec;
  MSI_TIME_STAMP_START(CreateInstanceFromRequest)
  auto status = GrpcTensorHelper::CreateInstanceFromRequest(request, &request_spec, &inputs);
  MSI_TIME_STAMP_END(CreateInstanceFromRequest)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "transfer request to instances failed";
    return status;
  }
  if (inputs.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Input instances count is 0";
  }

  MSI_TIME_STAMP_START(RUN_METHOD)
  status = RunAsync(request, reply, request_spec, inputs, callback);
  MSI_TIME_STAMP_END(RUN_METHOD)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "Run servable " << request_spec.Repr() << " failed";
    return status;
  }
  return SUCCESS;
}

Status Worker::RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                        const RequestSpec &request_spec, const std::vector<InstanceData> &inputs,
                        DispatchCallback callback) {
  const auto &worker = GetServableWorker(request_spec);
  if (worker.worker_service == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find servable match " << request_spec.Repr();
  }
  WorkCallBack on_process_done = [request, reply, callback](const std::vector<InstancePtr> &instances,
                                                            const Status &error_msg) {
    auto status = GrpcTensorHelper::CreateReplyFromInstances(request, instances, error_msg, reply);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "transfer result to reply failed";
    }
    callback(status);
  };
  (void)worker.worker_service->Work(request_spec, inputs, on_process_done);
  return SUCCESS;
}

void Worker::Update() {
  /*
  if (version_strategy_ == kVersionStrategySpecific) {
    return;
  }

  std::vector<uint64_t> versions;
  GetVersions(base_spec_, &versions);
  for (auto &version : versions) {
    bool isfind = std::any_of(work_list_.begin(), work_list_.end(), [&](const ServableWorkerContext &work) {
      return work.servable_spec.version_number == version;
    });
    if (isfind) {
      continue;
    }
    ServableWorkerContext work;
    LoadModel(&base_spec_, version, &work);
    auto status = AddWorker(work);
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "AddWorker failed";
    }
    work_list_.push_back(work);
    MSI_LOG_INFO << "Load Model version " << version << " success";
  }
  for (auto iter = work_list_.begin(); iter != work_list_.end();) {
    bool isfind = std::any_of(versions.begin(), versions.end(),
                              [&](const uint64_t &version) { return iter->servable_spec.version_number == version; });
    if (isfind) {
      ++iter;
      continue;
    }
    (void)RemoveWorker(*iter);
    session_->UnloadModel(iter->model_id);
    MSI_LOG_INFO << "UnLoad Model version " << iter->servable_spec.version_number << " success";
    work_list_.erase(iter);
  }
  */
}

Status Worker::StartGrpcServer(const std::shared_ptr<MSWorkerServer> &grpc_server, const std::string &worker_ip,
                               int32_t port) {
  if (worker_grpc_server_ != nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Worker gRPC server is already running";
  }
  worker_grpc_server_ = grpc_server;
  return worker_grpc_server_->StartWorkerGrpcServer(worker_ip, port);
}

Status Worker::StartServable(std::shared_ptr<ServableBase> servable, std::shared_ptr<BaseNotifyMaster> notify_master) {
  ExitSignalHandle::Instance().Start();  // handle ctrl+c to exit
  if (servable_started_) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }
  clear_flag_.clear();

  // start task queue for handle preprocess and postprocess
  py_task_queue_group_.Start();
  cpp_preprocess_.Start(2);
  cpp_postprocess_.Start(2);

  notify_master_ = std::move(notify_master);
  auto servable_name = servable->GetServableName();
  ServableSignature signature;
  if (!ServableStorage::Instance().GetServableDef(servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable " << servable_name << " has not been registered";
  }
  auto service = std::make_shared<WorkExecutor>(GetPyTaskQueuePreprocess(), GetPyTaskQueuePostprocess(),
                                                GetCppTaskQueuePreprocess(), GetCppTaskQueuePostprocess());
  auto status = service->Init(signature, servable);
  if (status != SUCCESS) {
    return status;
  }
  ServableWorkerContext work;
  WorkerSpec worker_spec;
  worker_spec.servable_name = servable_name;
  worker_spec.version_number = servable->GetServableVersion();
  for (auto &method : signature.methods) {
    WorkerMethodInfo worker_method_info;
    worker_method_info.name = method.method_name;
    for (auto &name : method.inputs) {
      worker_method_info.input_names.push_back(name);
    }
    worker_spec.methods.push_back(worker_method_info);
  }
  work.worker_spec = worker_spec;
  work.servable_signature = signature;
  work.worker_service = service;
  work.servable = servable;

  work_list_.push_back(work);

  status = RegisterWorker();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register worker failed";
    return status;
  }
  servable_started_ = true;
  return SUCCESS;
}

void Worker::StopServable(bool notify_master) {
  exit_notify_master_ = notify_master;
  ExitSignalHandle::Instance().Stop();
}

void Worker::Clear() {
  std::unique_lock<std::shared_mutex> lock(worker_shared_lock_);
  ServableStorage::Instance().Clear();
  worker_grpc_server_ = nullptr;
  if (clear_flag_.test_and_set()) {
    return;
  }
  MSI_LOG_INFO << "Start clear worker session";
  version_controller_.StopPollModelPeriodic();
  if (exit_notify_master_ && servable_started_) {
    notify_master_->Unregister();
  }
  for (auto &worker_item : work_list_) {
    worker_item.servable->Clear();
  }
  work_list_.clear();

  py_task_queue_group_.Stop();
  cpp_preprocess_.Stop();
  cpp_postprocess_.Stop();
  servable_started_ = false;
  MSI_LOG_INFO << "End clear worker session";
}

bool Worker::IsRunning() { return servable_started_; }

Worker::~Worker() { Clear(); }

ServableWorkerContext Worker::GetServableWorker(const RequestSpec &request_spec) {
  ServableWorkerContext context;
  if (request_spec.version_number != 0) {
    auto item = find_if(work_list_.begin(), work_list_.end(), [&](const ServableWorkerContext &v) {
      return v.worker_spec.servable_name == request_spec.servable_name &&
             v.worker_spec.version_number == request_spec.version_number;
    });
    if (item != work_list_.end()) {
      context = *item;
    }
  } else {
    uint64_t max_version = 0;
    for (auto &item : work_list_) {
      if (item.worker_spec.servable_name == request_spec.servable_name &&
          item.worker_spec.version_number > max_version) {
        context = item;
        max_version = item.worker_spec.version_number;
      }
    }
  }
  return context;
}

Worker::Worker() {}

size_t Worker::GetBatchSize() const {
  size_t batch_size_ret = 1;
  for (const auto &service : work_list_) {
    auto batch_size = service.servable->GetBatchSize();
    if (batch_size != 0) {
      batch_size_ret = batch_size;
      break;
    }
  }
  return batch_size_ret;
}

}  // namespace serving
}  // namespace mindspore
