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
#include <unistd.h>
#include <condition_variable>
#include <regex>
#include "pybind11/pybind11.h"
#include "common/proto_tensor.h"
#include "common/exit_handle.h"
#include "worker/context.h"
#include "worker/grpc/worker_process.h"
#include "worker/task_queue.h"
#include "worker/grpc/worker_server.h"
#include "worker/servable_register.h"

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
  worker_spec.servable_spec = servable_spec_;
  worker_spec.worker_address = worker_address;
  worker_spec.worker_pid = getpid();
  auto status = notify_master_->Register(worker_spec);
  return status;
}

Status Worker::RunAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                        const PredictOnFinish &on_finish) {
  Status status;
  RequestSpec request_spec;
  GrpcTensorHelper::GetRequestSpec(request, &request_spec);

  auto servable_name = request_spec.servable_name;
  auto method_name = request_spec.method_name;

  const ServableSignature &servable_signature = ServableRegister::Instance().GetServableSignature();
  if (servable_signature.servable_name != servable_name) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Servable " << servable_name << " is not declared";
  }
  auto method_signature = servable_signature.GetMethodDeclare(method_name);
  if (method_signature == nullptr) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << "Method " << method_name << " is not registered for servable " << servable_name;
  }
  const MethodSignature &method = *method_signature;
  std::vector<InstanceData> instances_data;
  status = GrpcTensorHelper::CreateInstanceFromRequest(method, request, &instances_data);
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "transfer request to instances failed";
    return status;
  }
  *(reply->mutable_servable_spec()) = request.servable_spec();
  WorkCallBack on_process_done = [&request, reply, on_finish, method](const std::vector<InstancePtr> &instances) {
    GrpcTensorHelper::CreateReplyFromInstances(request, method, instances, reply);
    on_finish();
  };
  return RunAsync(request_spec, instances_data, on_process_done);
}

Status Worker::RunAsync(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                        const WorkCallBack &on_process_done) {
  while (true) {
    // avoid deadlock when Worker::Clear->gRPC shutdown, while gRPC shutdown waiting all request finished
    if (worker_shared_lock_.try_lock_shared()) {
      auto status = RunAsyncInner(request_spec, instances_data, on_process_done);
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

Status Worker::RunAsyncInner(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                             const WorkCallBack &on_process_done) {
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(WORKER_UNAVAILABLE)
           << "RunAsync worker for inference failed, worker has not been started or stopped";
  }
  if (instances_data.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Input instances count is 0";
  }
  if (!CheckServableRequest(request_spec)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Cannot find servable match " << request_spec.Repr();
  }
  MSI_LOG_INFO << "New request, method: " << request_spec.method_name << ", instances count: " << instances_data.size();
  return worker_executor_.Work(request_spec, instances_data, on_process_done);
}

Status Worker::Run(const RequestSpec &request_spec, const std::vector<InstanceData> &instances_data,
                   std::vector<InstancePtr> *out) {
  if (!servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Run worker for inference failed, worker has not been started";
  }
  MSI_EXCEPTION_IF_NULL(out);
  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  WorkCallBack on_process_done = [promise, out](const std::vector<InstancePtr> &instances) {
    *out = instances;
    promise->set_value();
  };
  auto status = RunAsync(request_spec, instances_data, on_process_done);
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

Status Worker::StartDistributedGrpcServer(std::shared_ptr<DistributedModelLoader> servable,
                                          const std::string &server_address) {
  if (distributed_grpc_server_ != nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Distributed gRPC server is already running";
  }
  distributed_grpc_server_ = std::make_shared<DistributedWorkerGrpcServer>(servable, server_address);
  SSLConfig ssl_config;
  return distributed_grpc_server_->Start(server_address, ssl_config, gRpcMaxMBMsgSize, "Distributed gRPC");
}

Status Worker::StartServable(const std::string &servable_directory, const std::string &servable_name,
                             uint32_t version_number,
                             const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models,
                             const std::string &master_address, const std::string &worker_address, bool own_device) {
  auto status = StartServableInner(servable_name, version_number, models, own_device);
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
  status = INFER_STATUS(SUCCESS) << "Serving: Start servable success, servable directory: '" << servable_directory
                                 << "', servable name: '" << servable_name << "', version number: " << version_number;
  MSI_LOG_INFO << status.StatusMessage();
  std::cout << status.StatusMessage() << std::endl;
  return SUCCESS;
}

Status Worker::StartServableInner(const std::string &servable_name, uint32_t version_number,
                                  const std::map<std::string, std::shared_ptr<ModelLoaderBase>> &models,
                                  bool own_device) {
  if (servable_started_) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "A servable has been started, only one servable can run in a process currently.";
  }
  clear_flag_.clear();
  auto status = worker_executor_.Init(models);
  if (status != SUCCESS) {
    return status;
  }
  servable_spec_.servable_name = servable_name;
  servable_spec_.version_number = version_number;
  servable_spec_.batch_size = worker_executor_.GetMaxBatchSize();
  servable_spec_.methods.clear();
  servable_spec_.own_device = own_device;

  for (auto &model_it : models) {
    ModelInfo model_info;
    auto &model_key = model_it.first;
    auto &model = model_it.second;
    model_info.batch_size = model->GetBatchSize();
    auto graph_num = model->GetGraphNum();
    model_info.sub_graph_infos.resize(graph_num);
    for (uint64_t i = 0; i < graph_num; i++) {
      model_info.sub_graph_infos[i].input_infos = model->GetInputInfos(i);
      model_info.sub_graph_infos[i].output_infos = model->GetOutputInfos(i);
    }
    servable_spec_.models[model_key] = model_info;
  }
  const ServableSignature &signature = ServableRegister::Instance().GetServableSignature();
  for (auto &method : signature.methods) {
    ServableMethodInfo worker_method_info;
    bool has_model = false;
    bool has_func = false;
    for (auto &stage : method.stage_map) {
      if (stage.second.stage_type == kMethodStageTypeModel) {
        has_model = true;
      } else if (stage.second.stage_type == kMethodStageTypePyFunction ||
                 stage.second.stage_type == kMethodStageTypeCppFunction) {
        has_func = true;
      }
    }
    if (has_model && !has_func) {
      worker_method_info.only_model_stage = true;
    } else {
      worker_method_info.only_model_stage = false;
    }
    // This worker does not occupy device and is only used to run python function stage to support python parallelism.
    // If one method does not contain function stage, requests of this method do not need to routed to this
    // worker.
    if (!servable_spec_.own_device && worker_method_info.only_model_stage) {
      continue;
    }
    worker_method_info.name = method.method_name;
    for (auto &name : method.inputs) {
      worker_method_info.input_names.push_back(name);
    }
    servable_spec_.methods.push_back(worker_method_info);
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
  MSI_LOG_INFO << "Start clear worker session";
  servable_started_ = false;
  worker_executor_.Stop();
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
  MSI_LOG_INFO << "End clear worker session";
}

bool Worker::IsRunning() { return servable_started_; }

Worker::~Worker() {
  Clear();
  if (listening_parent_thread_.joinable()) {
    listening_parent_thread_.join();
  }
}

bool Worker::CheckServableRequest(const RequestSpec &request_spec) {
  if (servable_spec_.servable_name != request_spec.servable_name) {
    return false;
  }
  if (request_spec.version_number != 0 && servable_spec_.version_number != request_spec.version_number) {
    return false;
  }
  return true;
}

Worker::Worker() {}

void Worker::ClearOnSystemFailed(const Status &error_msg) {
  std::shared_lock<std::shared_mutex> lock(worker_shared_lock_);
  MSI_LOG_INFO << "Clear instances on system failed: " << error_msg.StatusMessage();
  worker_executor_.ClearInstances(error_msg);
}

static std::vector<int> GetAllChildrenPids(int cur_pid) {
  if (cur_pid <= 0) {
    return {};
  }
  std::string get_all_children_pids = "ps -o pid --no-headers --ppid " + std::to_string(cur_pid);
  FILE *fp = popen(get_all_children_pids.c_str(), "r");
  if (fp == nullptr) {
    return {};
  }
  constexpr int max_result_size = 1024;
  char buf[max_result_size] = {0};
  std::string cmd_result;
  while (fgets(buf, max_result_size, fp) != nullptr && cmd_result.size() <= max_result_size) {
    cmd_result += std::string(buf) + " ";
  }
  pclose(fp);
  if (cmd_result.size() == max_result_size || cmd_result.empty()) {
    return {};
  }
  std::regex pid_reg("[0-9]+");
  auto match_beg = std::sregex_iterator(cmd_result.begin(), cmd_result.end(), pid_reg);
  auto match_end = std::sregex_iterator();
  if (match_beg == match_end) {
    return {};
  }
  std::vector<int> direct_children;
  for (auto item = match_beg; item != match_end; ++item) {
    auto pid_str = item->str();
    auto pid = static_cast<int>(std::strtol(pid_str.c_str(), nullptr, 10));
    if (pid <= 0) {
      continue;
    }
    std::ifstream stat_fp("/proc/" + std::to_string(pid) + "/stat");
    if (!stat_fp.is_open()) {
      continue;
    }
    constexpr int cache_size_max = 128;
    char cache[cache_size_max + 1] = {0};
    stat_fp.read(cache, cache_size_max);
    std::string cache_str = cache;
    auto pos = cache_str.find(") ");
    if (pos == std::string::npos) {
      continue;
    }
    cache_str = cache_str.substr(pos + strlen(") S "));
    int child_ppid = static_cast<int>(std::strtol(cache_str.c_str(), nullptr, 10));
    if (child_ppid != cur_pid) {
      continue;
    }
    direct_children.push_back(pid);
  }
  std::vector<int> all_pids = direct_children;
  for (auto &pid : direct_children) {
    auto pids = GetAllChildrenPids(pid);
    all_pids.insert(all_pids.end(), pids.begin(), pids.end());
  }
  return all_pids;
}

void Worker::StartListeningParentExitThread() {
  auto thread_func = [this]() {
    MSI_LOG_INFO << "Start listening parent";
    auto init_parent_pid = getppid();
    constexpr int sleep_period_in_ms = 100;
    constexpr int try_kill_children_times = 100;
    // exit when receive SIGINT SIGTERM, or parent process exit
    while (true) {
      if (ExitSignalHandle::Instance().HasStopped()) {
        MSI_LOG_WARNING << "Worker has received exit message, worker begin to exit";
        break;
      }
      auto cur_parent_pid = getppid();
      if (init_parent_pid != cur_parent_pid) {
        MSI_LOG_WARNING << "Worker detect parent pid=" << init_parent_pid << " has exited, worker begin to exit";
        ExitSignalHandle::Instance().Stop();
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_period_in_ms));
    }
    Clear();
    auto cur_pid = getpid();
    for (int i = 0; i < try_kill_children_times; i++) {  // 100*100ms=10s
      auto child_pids = GetAllChildrenPids(cur_pid);
      if (child_pids.empty() && !continue_listen_children_) {
        break;
      }
      for (auto pid : child_pids) {
        kill(pid, SIGTERM);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_period_in_ms));
    }
    MSI_LOG_INFO << "Stop listening parent";
  };
  listening_parent_thread_ = std::thread(thread_func);
}
}  // namespace serving
}  // namespace mindspore
