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
#include "pybind11/pybind11.h"
#include "common/proto_tensor.h"
#include "common/file_system_operation.h"
#include "common/exit_handle.h"
#include "worker/context.h"
#include "master/server.h"
#include "worker/grpc/worker_process.h"
#include "worker/task_queue.h"

namespace py = pybind11;

namespace mindspore {
namespace serving {

static const char *kVersionStrategyLastest = "lastest";
static const char *kVersionStrategySpecific = "specific";

Worker &Worker::GetInstance() {
  static Worker instance;
  ExitHandle::Instance().InitSignalHandle();
  return instance;
}

Status Worker::StartGrpcServer(const std::string &ip, uint32_t grpc_port) {
  return grpc_server_.Start(std::make_shared<MSWorkerImpl>(), ip, grpc_port, gRpcMaxMBMsgSize, "Worker gRPC");
}

Status Worker::RegisterWorker() {
  std::vector<LoadServableSpec> specs;
  std::vector<ServableSignature> signatures;
  for (auto &work : work_list_) {
    specs.push_back(work.servable_spec);
    signatures.push_back(work.servable_signature);
  }
  std::vector<WorkerSpec> worker_specs;
  for (size_t i = 0; i < specs.size(); i++) {
    auto &spec = specs[i];
    auto &servable_signature = signatures[i];
    WorkerSpec worker_spec;
    worker_spec.servable_name = spec.servable_name;
    worker_spec.version_number = spec.version_number;
    for (auto &method : servable_signature.methods) {
      WorkerMethodInfo worker_method_info;
      worker_method_info.name = method.method_name;
      for (auto &name : method.inputs) {
        worker_method_info.input_names.push_back(name);
      }
      worker_spec.methods.push_back(worker_method_info);
    }
    worker_specs.push_back(worker_spec);
  }
  auto status = notify_master_->Register(worker_specs);
  return status;
}

Status Worker::StartVersionController() {
  version_controller_.StartPollModelPeriodic();
  return SUCCESS;
}

Status Worker::AddWorker(const ServableWorkerContext &work) {
  WorkerSpec worker_spec;
  worker_spec.servable_name = work.servable_spec.servable_name;
  worker_spec.version_number = work.servable_spec.version_number;
  for (auto &method : work.servable_signature.methods) {
    WorkerMethodInfo worker_method_info;
    worker_method_info.name = method.method_name;
    for (auto &name : method.inputs) {
      worker_method_info.input_names.push_back(name);
    }
    worker_spec.methods.push_back(worker_method_info);
  }
  return notify_master_->AddWorker(worker_spec);
}

Status Worker::RemoveWorker(const ServableWorkerContext &work) {
  WorkerSpec worker_spec;
  worker_spec.servable_name = work.servable_spec.servable_name;
  worker_spec.version_number = work.servable_spec.version_number;
  for (auto &method : work.servable_signature.methods) {
    WorkerMethodInfo worker_method_info;
    worker_method_info.name = method.method_name;
    for (auto &name : method.inputs) {
      worker_method_info.input_names.push_back(name);
    }
    worker_spec.methods.push_back(worker_method_info);
  }
  return notify_master_->RemoveWorker(worker_spec);
}

Status Worker::Run(const proto::PredictRequest &request, proto::PredictReply *reply) {
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
  std::vector<Instance> outputs;
  MSI_TIME_STAMP_START(RUN_METHOD)
  status = Run(request_spec, inputs, &outputs);
  MSI_TIME_STAMP_END(RUN_METHOD)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "Run servable " << request_spec.Repr() << " failed";
    return status;
  }
  MSI_TIME_STAMP_START(CreateReplyFromInstances)
  status = GrpcTensorHelper::CreateReplyFromInstances(request, outputs, reply);
  MSI_TIME_STAMP_END(CreateReplyFromInstances)
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "transfer result to reply failed";
    return status;
  }
  MSI_LOG(INFO) << "run Predict finished";
  return SUCCESS;
}

Status Worker::Run(const RequestSpec &request_spec, const std::vector<serving::InstanceData> &inputs,
                   std::vector<serving::Instance> *outputs) {
  MSI_EXCEPTION_IF_NULL(outputs);
  auto result_pair = RunAsync(request_spec, inputs);
  if (result_pair.first != SUCCESS) {
    return result_pair.first;
  }
  auto result = result_pair.second;
  while (result->HasNext()) {
    serving::Instance instance;
    result->GetNext(&instance);
    outputs->push_back(instance);
  }
  return SUCCESS;
}

std::pair<Status, std::shared_ptr<AsyncResult>> Worker::RunAsync(const RequestSpec &request_spec,
                                                                 const std::vector<InstanceData> &inputs) {
  std::shared_ptr<AsyncResult> result = std::make_shared<AsyncResult>(inputs.size());
  const auto &worker = GetServableWorker(request_spec);
  if (worker.worker_service == nullptr) {
    return {INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find servable match " << request_spec.Repr(), nullptr};
  }
  WorkCallBack on_process_done = [result](const Instance &output, const Status &error_msg) {
    auto output_index = output.context.instance_index;
    if (output_index < result->result_.size()) {
      result->result_[output_index].error_msg = error_msg;
      if (error_msg == SUCCESS) {
        result->result_[output_index] = output;
      }
    }
  };
  result->future_list_ = worker.worker_service->Work(request_spec, inputs, on_process_done);
  return {SUCCESS, result};
}

Status Worker::InitEnv(ModelType model_type, const std::unordered_map<std::string, std::string> &other_options) {
  Status status;
  if (session_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Session has been inited";
  }
  auto context = ServableContext::Instance();
  DeviceType device_type = kDeviceTypeNotSpecified;
  session_ = InferSessionStorage::Instance().Get(context->GetDeviceType(), model_type, &device_type);
  if (session_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Cannot find session registed for device type " << context->GetDeviceType();
  }
  if (device_type != kDeviceTypeNotSpecified) {
    context->SetDeviceType(device_type);
  }
  status = session_->InitEnv(context->GetDeviceType(), context->GetDeviceId(), other_options);
  if (status != SUCCESS) {
    session_ = nullptr;
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Init env failed, device type " << context->GetDeviceType() << ", device id " << context->GetDeviceId();
  }
  return SUCCESS;
}

Status Worker::FinalizeEnv() {
  if (session_ != nullptr) {
    return session_->FinalizeEnv();
  }
  return SUCCESS;
}
Status Worker::LoadModel(LoadServableSpec *servable_spec, uint64_t version_number, ServableWorkerContext *work) {
  MSI_EXCEPTION_IF_NULL(servable_spec);
  MSI_EXCEPTION_IF_NULL(work);
  servable_spec->version_number = version_number;
  ServableSignature signature;
  if (!ServableStorage::Instance()->GetServableDef(servable_spec->servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable " << servable_spec->servable_name << " has not been registerd";
  }
  const auto &servable_meta = signature.servable_meta;
  std::string model_file_name = servable_spec->servable_directory + "/" + servable_spec->servable_name + "/" +
                                std::to_string(version_number) + "/" + servable_meta.servable_file;
  uint32_t model_id;
  auto context = ServableContext::Instance();
  Status status = session_->LoadModelFromFile(context->GetDeviceType(), context->GetDeviceId(), model_file_name,
                                              servable_meta.model_format, &model_id);
  if (status != SUCCESS) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Load model failed, servable directory: '" << servable_spec->servable_directory << "', servable name: '"
           << servable_spec->servable_name << "', servable file: '" << servable_meta.servable_file
           << "', version number " << version_number;
  }
  auto service = std::make_shared<WorkExecutor>(GetPyTaskQueuePreprocess(), GetPyTaskQueuePostprocess(),
                                                GetCppTaskQueuePreprocess(), GetCppTaskQueuePostprocess());
  status = service->Init(signature, std::make_shared<AscendModelServable>(session_, model_id));
  if (status != SUCCESS) {
    return status;
  }
  work->servable_spec = *servable_spec;
  work->servable_signature = signature;
  work->worker_service = service;
  work->model_id = model_id;
  work->model_file_name = model_file_name;
  return SUCCESS;
}

void Worker::Update() {
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
}

Status Worker::StartServable(const std::string &servable_directory, const std::string &servable_name,
                             uint32_t version_number, std::shared_ptr<BaseNotifyMaster> notify_master) {
  if (servable_started_) {
    MSI_LOG_EXCEPTION << "A servable has been started, only one servable can run in a process currently.";
  }
  notify_master_ = std::move(notify_master);
  base_spec_.servable_directory = servable_directory;
  base_spec_.servable_name = servable_name;
  base_spec_.version_number = version_number;

  std::string version_strategy;
  if (version_number == 0) {
    version_strategy = kVersionStrategyLastest;
  } else {
    version_strategy = kVersionStrategySpecific;
  }
  Status status;
  ServableSignature signature;
  if (!ServableStorage::Instance()->GetServableDef(servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable '" << servable_name << "' has not been registered";
  }
  if (session_ == nullptr) {
    status = InitEnv(signature.servable_meta.model_format, {});
    if (status != SUCCESS) {
      MSI_LOG_ERROR << "Init env failed";
      return status;
    }
  }

  std::vector<uint64_t> real_versions;
  status = LoadServableConfig(base_spec_, version_strategy, &real_versions);
  if (status != SUCCESS) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Start servable failed, there is no servable of the specified version number, specified version number: "
           << version_number << ", servable directory: '" << base_spec_.servable_directory << "', servable name: '"
           << base_spec_.servable_name
           << "'. version number is a positive integer(started from 1) and 0 represents the maximum version number.";
  }
  for (auto real_version_number : real_versions) {
    ServableWorkerContext work;
    status = LoadModel(&base_spec_, real_version_number, &work);
    if (status != SUCCESS) {
      return status;
    }
    work_list_.push_back(work);
  }
  status = RegisterWorker();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register worker failed";
    return status;
  }
  servable_started_ = true;
  status = INFER_STATUS(SUCCESS) << "Serving: Start servable success, servable directory: '" << servable_directory
                                 << "', servable name: '" << servable_name
                                 << "', specified version number: " << version_number
                                 << ", started version numbers: " << real_versions;
  MSI_LOG_INFO << status.StatusMessage();
  std::cout << status.StatusMessage() << std::endl;
  return SUCCESS;
}

void Worker::StopServable(bool notify_master) {
  exit_notify_master_ = notify_master;
  ExitHandle::Instance().Stop();
}

void Worker::Clear() {
  if (clear_flag_.test_and_set()) {
    return;
  }
  MSI_LOG_INFO << "Start clear worker session";
  servable_stoppedd_ = true;
  version_controller_.StopPollModelPeriodic();
  if (exit_notify_master_ && servable_started_) {
    notify_master_->Unregister();
  }
  if (session_ != nullptr) {
    for (auto &it : work_list_) {
      session_->UnloadModel(it.model_id);
    }
  }
  work_list_.clear();
  FinalizeEnv();

  session_ = nullptr;
  py_task_queue_group_.Stop();
  cpp_preprocess_.Stop();
  cpp_postprocess_.Stop();
  grpc_server_.Stop();
  MSI_LOG_INFO << "End clear worker session";
}

bool Worker::HasCleared() { return servable_stoppedd_; }

Worker::~Worker() { Clear(); }

void Worker::GetVersions(const LoadServableSpec &servable_spec, std::vector<uint64_t> *real_versions) {
  MSI_EXCEPTION_IF_NULL(real_versions);
  // define version_strategy:"specific","lastest","multi"
  if (version_strategy_ == kVersionStrategySpecific) {
    real_versions->push_back(servable_spec.version_number);
    return;
  }
  auto trans_to_integer = [](const std::string &str) -> uint32_t {
    uint32_t parsed_value = 0;
    for (auto c : str) {
      if (c < '0' || c > '9') {
        return 0;
      }
      parsed_value = parsed_value * 10 + c - '0';
    }
    if (std::to_string(parsed_value) != str) {
      return 0;
    }
    return parsed_value;
  };
  uint64_t newest_version = 0;
  std::string model_path = servable_spec.servable_directory + "/" + servable_spec.servable_name;
  auto sub_dir = GetAllSubDirsNotFullPath(model_path);
  static std::set<std::string> ignore_dir;
  for (const auto &dir : sub_dir) {
    if (dir == "__pycache__") continue;
    auto version_parse = trans_to_integer(dir);
    if (version_parse == 0) {
      if (ignore_dir.emplace(servable_spec.servable_directory + dir).second) {
        MSI_LOG_INFO << "Ignore directory " << dir << ", model_directory " << servable_spec.servable_directory
                     << ", model_name " << servable_spec.servable_name;
      }
      continue;
    }
    real_versions->push_back(version_parse);
    if (version_parse > newest_version) {
      newest_version = version_parse;
    }
  }
  if (version_strategy_ == kVersionStrategyLastest) {
    real_versions->clear();
    if (newest_version != 0) {
      real_versions->push_back(newest_version);
    }
  }
}
Status Worker::LoadServableConfig(const LoadServableSpec &servable_spec, const std::string &version_strategy,
                                  std::vector<uint64_t> *real_versions) {
  MSI_EXCEPTION_IF_NULL(real_versions);
  auto model_directory = servable_spec.servable_directory;
  auto model_name = servable_spec.servable_name;

  if (!DirOrFileExist(model_directory + "/" + model_name)) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Model not found, model_directory " << model_directory << ", model_name " << model_name;
  }
  std::string model_path = model_directory + "/" + model_name;
  auto version_directory = [model_path](int64_t version_number) {
    return model_path + "/" + std::to_string(version_number);
  };
  version_strategy_ = version_strategy;
  // version_strategy:"specific","lastest","multi"
  GetVersions(servable_spec, real_versions);
  if (real_versions->size() == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Not found invalid model version , model_directory " << model_directory << ", model_name " << model_name;
  }
  for (auto real_version_number : *real_versions) {
    if (!DirOrFileExist(version_directory(real_version_number))) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Open failed for version " << real_version_number << ", model_directory "
                                            << model_directory << ", model_name " << model_name;
    }
  }
  return SUCCESS;
}

ServableWorkerContext Worker::GetServableWorker(const RequestSpec &request_spec) {
  ServableWorkerContext context;
  if (request_spec.version_number != 0) {
    auto item = find_if(work_list_.begin(), work_list_.end(), [&](const ServableWorkerContext &v) {
      return v.servable_spec.servable_name == request_spec.servable_name &&
             v.servable_spec.version_number == request_spec.version_number;
    });
    if (item != work_list_.end()) {
      context = *item;
    }
  } else {
    uint64_t max_version = 0;
    for (auto &item : work_list_) {
      if (item.servable_spec.servable_name == request_spec.servable_name &&
          item.servable_spec.version_number > max_version) {
        context = item;
        max_version = item.servable_spec.version_number;
      }
    }
  }
  return context;
}

Worker::Worker() {
  cpp_preprocess_.Start(2);
  cpp_postprocess_.Start(2);
}

ssize_t Worker::GetBatchSize() const {
  ssize_t batch_size_ret = -1;
  for (auto service : work_list_) {
    auto batch_size = session_->GetBatchSize(service.model_id);
    if (batch_size != -1) {
      batch_size_ret = batch_size;
      break;
    }
  }
  return batch_size_ret;
}

AsyncResult::AsyncResult(size_t size) : result_(size), next_index_(0) {}

bool AsyncResult::HasNext() { return next_index_ < future_list_.size(); }

Status AsyncResult::GetNext(Instance *instance_result) {
  MSI_EXCEPTION_IF_NULL(instance_result);
  if (next_index_ >= future_list_.size()) {
    MSI_LOG_ERROR << "GetNext failed, index greater than instance count " << future_list_.size();
    return FAILED;
  }
  auto index = next_index_;
  next_index_++;
  auto &future = future_list_[index];
  if (!future.valid()) {
    instance_result->error_msg = result_[index].error_msg;
    return FAILED;
  }
  const int kWaitMaxHundredMs = 100;
  int i;
  for (i = 0; i < kWaitMaxHundredMs; i++) {  //
    if (Worker::GetInstance().HasCleared()) {
      instance_result->error_msg = Status(FAILED, "Servable stopped");
      return FAILED;
    }
    if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      break;
    }
  }
  if (i >= kWaitMaxHundredMs) {
    MSI_LOG_ERROR << "GetNext failed, wait time out, index " << index << ", total count " << future_list_.size();
    instance_result->error_msg = Status(FAILED, "Time out");
    return FAILED;
  }

  future.get();
  *instance_result = result_[index];
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
