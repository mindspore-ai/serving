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

Status Worker::RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "RunAsync worker for inference failed, worker has not been started";
  }
  MSI_EXCEPTION_IF_NULL(reply);
  std::vector<InstanceData> instances_data;
  RequestSpec request_spec;
  auto status = GrpcTensorHelper::CreateInstanceFromRequest(request, &request_spec, &instances_data);
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "transfer request to instances failed";
    return status;
  }
  if (instances_data.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Input instances count is 0";
  }

  const auto &worker = GetServableWorker(request_spec);
  if (worker.worker_service == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find servable match " << request_spec.Repr();
  }
  WorkCallBack on_process_done = [request, reply, on_finish](const std::vector<InstancePtr> &instances) {
    GrpcTensorHelper::CreateReplyFromInstances(request, instances, reply);
    on_finish();
  };
  return worker.worker_service->Work(request_spec, instances_data, on_process_done);
}

void Worker::Update() {}

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
