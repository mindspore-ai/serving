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
#include <algorithm>
#include <condition_variable>
#include "pybind11/pybind11.h"
#include "common/proto_tensor.h"
#include "common/file_system_operation.h"
#include "common/exit_handle.h"
#include "worker/context.h"
#include "worker/grpc/worker_process.h"
#include "worker/task_queue.h"
#include "worker/grpc/worker_server.h"
#include "worker/pipeline.h"

namespace py = pybind11;

namespace mindspore {
namespace serving {

Worker &Worker::GetInstance() {
  static Worker instance;
  return instance;
}

Status Worker::RegisterWorker(const std::string &master_address, const std::string &worker_address) {
  notify_master_ = std::make_shared<GrpcNotifyMaster>(master_address, worker_address);
  WorkerRegSpec worker_spec;
  if (pipeline_spec_.methods.empty()) {
    worker_spec.servable_spec = servable_context_.servable_spec;
  } else {
    worker_spec.servable_spec = pipeline_spec_;
  }
  auto status = notify_master_->Register(worker_spec);
  return status;
}

Status Worker::RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply, PredictOnFinish on_finish) {
  while (true) {
    if (worker_shared_lock_.try_lock_shared()) {
      auto status = RunAsyncInner(request, reply, on_finish);
      worker_shared_lock_.unlock_shared();
      return status;
    } else if (!servable_started_) {
      return INFER_STATUS_LOG_ERROR(WORKER_UNAVAILABLE)
             << "RunAsync worker for inference failed, worker has not been started or stopped";
    }
    std::chrono::milliseconds duration(1);  // 1ms
    std::this_thread::sleep_for(duration);
  }
}

Status Worker::RunPipeline(const proto::PredictRequest &request, proto::PredictReply *reply,
                           PredictOnFinish on_finish) {
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "RunPipeline worker for inference failed, worker has not been started";
  }
  MSI_EXCEPTION_IF_NULL(reply);
  std::vector<InstanceData> instances_data;
  RequestSpec request_spec;
  auto status = GrpcTensorHelper::CreatePipelineInstanceFromRequest(request, &request_spec, &instances_data);
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "transfer request to instances failed";
    return status;
  }
  if (instances_data.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Input instances count is 0";
  }

  WorkCallBack on_process_done = [request, reply, on_finish](const std::vector<InstancePtr> &instances) {
    GrpcTensorHelper::CreateReplyFromInstances(request, instances, reply);
    on_finish();
  };
  PipelineSignature method_signature;
  PipelineStorage::Instance().GetMethodDeclare(request_spec.method_name, &method_signature);
  return servable_context_.worker_service->Pipe(request_spec, instances_data, method_signature, on_process_done);
}

Status Worker::RunAsyncInner(const proto::PredictRequest &request, proto::PredictReply *reply,
                             PredictOnFinish on_finish) {
  if (!pipeline_spec_.methods.empty()) {
    return RunPipeline(request, reply, on_finish);
  }
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(WORKER_UNAVAILABLE)
           << "RunAsync worker for inference failed, worker has not been started or stopped";
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
  if (!CheckServableRequest(request_spec)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find servable match " << request_spec.Repr();
  }
  WorkCallBack on_process_done = [request, reply, on_finish](const std::vector<InstancePtr> &instances) {
    GrpcTensorHelper::CreateReplyFromInstances(request, instances, reply);
    on_finish();
  };
  return servable_context_.worker_service->Work(request_spec, instances_data, on_process_done);
}

Status Worker::Run(const RequestSpec &request_spec, const std::vector<InstanceData> instances_data,
                   std::vector<InstancePtr> *out) {
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Run worker for inference failed, worker has not been started";
  }
  MSI_EXCEPTION_IF_NULL(out);
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  WorkCallBack on_process_done = [promise, out](const std::vector<InstancePtr> &instances) {
    std::copy(instances.begin(), instances.end(), std::back_inserter(*out));
    promise->set_value();
  };
  auto status = servable_context_.worker_service->Work(request_spec, instances_data, on_process_done);
  if (status != SUCCESS) {
    return status;
  }
  future.get();
  return SUCCESS;
}

Status Worker::StartGrpcServer(const std::string &server_address) {
  if (worker_grpc_server_ != nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Worker gRPC server is already running";
  }
  worker_grpc_server_ = std::make_shared<WorkerGrpcServer>();
  SSLConfig ssl_config;
  return worker_grpc_server_->Start(server_address, ssl_config, gRpcMaxMBMsgSize, "Worker gRPC");
}

Status Worker::StartDistributedGrpcServer(std::shared_ptr<DistributedServable> servable,
                                          const std::string &server_address) {
  if (distributed_grpc_server_ != nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Distributed gRPC server is already running";
  }
  distributed_grpc_server_ = std::make_shared<DistributedWorkerGrpcServer>(servable, server_address);
  SSLConfig ssl_config;
  return distributed_grpc_server_->Start(server_address, ssl_config, gRpcMaxMBMsgSize, "Distributed gRPC");
}
void Worker::InitPipeline(const std::string &servable_name, uint64_t version_number) {
  std::vector<PipelineSignature> pipelines;
  if (PipelineStorage::Instance().GetPipelineDef(&pipelines)) {
    pipeline_spec_.servable_name = servable_name;
    pipeline_spec_.version_number = version_number;
    pipeline_spec_.batch_size = 1;
    for (auto &method : pipelines) {
      ServableMethodInfo worker_method_info;
      worker_method_info.name = method.pipeline_name;
      for (auto &name : method.inputs) {
        worker_method_info.input_names.push_back(name);
      }
      pipeline_spec_.methods.push_back(worker_method_info);
    }
  }
}

Status Worker::StartServable(const std::shared_ptr<ServableBase> &servable, const std::string &master_address,
                             const std::string &worker_address) {
  auto status = StartServableInner(servable);
  if (status != SUCCESS) {
    return status;
  }
  status = StartGrpcServer(worker_address);
  if (status != SUCCESS) {
    return status;
  }
  status = RegisterWorker(master_address, worker_address);
  if (status != SUCCESS) {
    return status;
  }
  status = INFER_STATUS(SUCCESS) << "Serving: Start servable success, servable directory: '"
                                 << servable->GetServableDirectory() << "', servable name: '"
                                 << servable->GetServableName()
                                 << "', version number: " << servable->GetServableVersion();
  MSI_LOG_INFO << status.StatusMessage();
  std::cout << status.StatusMessage() << std::endl;
  return SUCCESS;
}

Status Worker::StartServableInner(const std::shared_ptr<ServableBase> &servable) {
  if (servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "A servable has been started, only one servable can run in a process currently.";
  }
  clear_flag_.clear();

  // start task queue for handle preprocess and postprocess
  py_task_queue_group_.Start();
  cpp_preprocess_.Start(2);
  cpp_postprocess_.Start(2);

  auto servable_name = servable->GetServableName();
  ServableSignature signature;
  if (!ServableStorage::Instance().GetServableDef(servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable " << servable_name << " has not been registered";
  }
  auto service =
    std::make_shared<WorkExecutor>(GetPyTaskQueuePreprocess(), GetPyTaskQueuePostprocess(), GetCppTaskQueuePreprocess(),
                                   GetCppTaskQueuePostprocess(), GetPyTaskQueuePipeline());
  auto status = service->Init(signature, servable);
  if (status != SUCCESS) {
    return status;
  }
  ServableWorkerContext servable_context;
  ServableRegSpec &servable_spec = servable_context.servable_spec;
  servable_spec.servable_name = servable_name;
  servable_spec.version_number = servable->GetServableVersion();
  servable_spec.batch_size = servable->GetBatchSize();
  if (servable_spec.batch_size == 0) {  // with_batch_size=False
    servable_spec.batch_size = 1;
  }
  for (auto &method : signature.methods) {
    ServableMethodInfo worker_method_info;
    worker_method_info.name = method.method_name;
    for (auto &name : method.inputs) {
      worker_method_info.input_names.push_back(name);
    }
    servable_spec.methods.push_back(worker_method_info);
  }
  servable_context.servable_signature = signature;
  servable_context.worker_service = service;
  servable_context.servable = servable;
  servable_context_ = servable_context;
  InitPipeline(servable_spec.servable_name, servable_spec.version_number);
  servable_started_ = true;
  return SUCCESS;
}

void Worker::StopServable(bool notify_master) {
  exit_notify_master_ = notify_master;
  ExitSignalHandle::Instance().Stop();
}

void Worker::Clear() {
  std::unique_lock<std::shared_mutex> lock(worker_shared_lock_);
  MSI_LOG_INFO << "Start clear worker session";
  servable_started_ = false;
  if (servable_context_.servable) {
    servable_context_.servable->Clear();
    servable_context_.servable = nullptr;
  }
  if (servable_context_.worker_service) {
    servable_context_.worker_service = nullptr;
  }
  if (exit_notify_master_ && notify_master_) {
    notify_master_->Unregister();
  }
  if (worker_grpc_server_) {
    worker_grpc_server_->Stop();
    worker_grpc_server_ = nullptr;
  }
  if (distributed_grpc_server_) {
    distributed_grpc_server_->Stop();
    distributed_grpc_server_ = nullptr;
  }
  py_task_queue_group_.Stop();
  cpp_preprocess_.Stop();
  cpp_postprocess_.Stop();

  MSI_LOG_INFO << "End clear worker session";
}

bool Worker::IsRunning() { return servable_started_; }

Worker::~Worker() { Clear(); }

bool Worker::CheckServableRequest(const RequestSpec &request_spec) {
  auto &servable_spec = servable_context_.servable_spec;
  if (servable_spec.servable_name != request_spec.servable_name) {
    return false;
  }
  if (request_spec.version_number != 0 && servable_spec.version_number != request_spec.version_number) {
    return false;
  }
  return true;
}

Worker::Worker() {}

size_t Worker::GetBatchSize() const {
  if (!servable_context_.servable) {
    return 0;
  }
  return servable_context_.servable->GetBatchSize();
}

void Worker::PushPyPreprocessResult(std::vector<ResultInstance> outputs) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  if (!servable_started_) {
    MSI_LOG_INFO << "Worker has not started or has exited";
    return;
  }
  GetPyTaskQueuePreprocess()->PushTaskPyResult(outputs);
}

void Worker::PushPyPostprocessResult(std::vector<ResultInstance> outputs) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  if (!servable_started_) {
    MSI_LOG_INFO << "Worker has not started or has exited";
    return;
  }
  GetPyTaskQueuePostprocess()->PushTaskPyResult(outputs);
}

void Worker::ClearOnSystemFailed(const Status &error_msg) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  if (servable_context_.worker_service) {
    MSI_LOG_INFO << "Clear instances on system failed: " << error_msg.StatusMessage();
    servable_context_.worker_service->ClearInstances(error_msg);
  }
}
void Worker::PushPyPipelineResult(std::vector<ResultInstance> outputs) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  if (!servable_started_) {
    MSI_LOG_INFO << "Worker has not started or has exited";
    return;
  }
  GetPyTaskQueuePipeline()->PushTaskPyResult(outputs);
}

}  // namespace serving
}  // namespace mindspore
