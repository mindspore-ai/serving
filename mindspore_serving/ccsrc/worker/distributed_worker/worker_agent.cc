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
#include "worker/distributed_worker/worker_agent.h"
#include <memory>
#include <string>
#include "worker/distributed_worker/agent_process/agent_process.h"
#include "worker/distributed_worker/notify_distributed/notify_worker.h"
#include "common/exit_handle.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {
WorkerAgent &WorkerAgent::Instance() {
  static WorkerAgent instance;
  return instance;
}

Status WorkerAgent::Clear() {
  if (notify_worker_) {
    if (exit_notify_worker_) {
      notify_worker_->Unregister();
      MSI_LOG_INFO << "End unregister to worker";
    }
    notify_worker_ = nullptr;
  }
  grpc_server_.Stop();
  if (session_ != nullptr) {
    session_->UnloadModel();
    session_ = nullptr;
  }
  return SUCCESS;
}

Status WorkerAgent::StartAgent(const AgentStartUpConfig &config, const std::string &dec_key,
                               const std::string &dec_mode) {
  session_ = InferenceLoader::Instance().CreateMindSporeInfer();
  if (session_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Create MindSpore infer failed";
  }
  Status status;
  config_ = config;
  const auto &common_meta = config.common_meta;
  auto enable_lite = InferenceLoader::Instance().GetEnableLite();
  status = session_->LoadModelFromFile(kDeviceTypeAscend, config.device_id, config.model_file_names, kMindIR,
                                       common_meta.with_batch_dim, common_meta.without_batch_dim_inputs, ModelContext(),
                                       dec_key, dec_mode, {}, enable_lite);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "LoadModelFromFile failed, servable name: " << common_meta.servable_name
                  << ", rank_id: " << config.rank_id << ", device id: " << config.device_id
                  << ", model file: " << config.model_file_names
                  << ", rank table file: " << config.rank_table_json_file_name
                  << ", group config file: " << config.group_file_names;
    return status;
  }
  status = StartGrpcServer();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Start agent grpc server failed, agent address: " << config.agent_address;
    return status;
  }
  status = RegisterAgent();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Register agent failed, agent address: " << config.agent_address
                  << ", distributed worker address: " << config.distributed_address;
    return status;
  }
  MSI_LOG_INFO << "Start agent success, servable name: " << common_meta.servable_name << ", rank_id: " << config.rank_id
               << ", device id: " << config.device_id << ", model file: " << config.model_file_names
               << ", rank table file: " << config.rank_table_json_file_name
               << ", group config file: " << config.group_file_names;
  return SUCCESS;
}

Status WorkerAgent::StartGrpcServer() {
  std::string server_address = config_.agent_address;
  return grpc_server_.Start(std::make_shared<MSAgentImpl>(server_address), server_address, gRpcMaxMBMsgSize, "Agent");
}

Status WorkerAgent::RegisterAgent() {
  notify_worker_ = std::make_shared<GrpcNotifyDistributeWorker>(config_.distributed_address, config_.agent_address);
  auto graph_num = session_->GetSubGraphNum();
  if (graph_num == 0) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "RegisterAgent failed, Agent graph_num error";
  }
  std::vector<WorkerAgentSpec> worker_specs;
  for (uint64_t i = 0; i < graph_num; i++) {
    WorkerAgentSpec spec;
    spec.subgraph = i;
    spec.agent_address = config_.agent_address;
    spec.rank_id = config_.rank_id;
    spec.batch_size = session_->GetBatchSize(i);
    spec.input_infos = session_->GetInputInfos(i);
    spec.output_infos = session_->GetOutputInfos(i);
    worker_specs.push_back(spec);
  }
  return notify_worker_->Register(worker_specs);
}

void WorkerAgent::StopAgent(bool notify_worker) {
  exit_notify_worker_ = notify_worker;
  ExitSignalHandle::Instance().Stop();
}

class ProtoDistributedPredictRequest : public RequestBase {
 public:
  explicit ProtoDistributedPredictRequest(const proto::DistributedPredictRequest &other) : proto_request_(other) {
    for (int i = 0; i < proto_request_.inputs_size(); i++) {
      (void)tensor_list_.emplace_back(const_cast<proto::Tensor *>(&proto_request_.inputs(i)));
    }
  }
  ~ProtoDistributedPredictRequest() = default;

  size_t size() const override { return tensor_list_.size(); }
  const TensorBase *operator[](size_t index) const override {
    if (index >= tensor_list_.size()) {
      MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
    }
    return &tensor_list_[index];
  }

 private:
  std::vector<ProtoTensor> tensor_list_;
  const proto::DistributedPredictRequest &proto_request_;
};

class ProtoDistributedPredictReply : public ReplyBase {
 public:
  explicit ProtoDistributedPredictReply(proto::DistributedPredictReply *other) : proto_reply_(other) {}
  ~ProtoDistributedPredictReply() = default;

  size_t size() const override { return tensor_list_.size(); };
  TensorBase *operator[](size_t index) override {
    if (index >= tensor_list_.size()) {
      MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
    }
    return &tensor_list_[index];
  };
  const TensorBase *operator[](size_t index) const override {
    if (index >= tensor_list_.size()) {
      MSI_LOG_EXCEPTION << "visit invalid index " << index << " total size " << tensor_list_.size();
    }
    return &tensor_list_[index];
  }
  TensorBase *add() override {
    auto tensor = proto_reply_->add_outputs();
    ProtoTensor proto_tensor(tensor);
    tensor_list_.push_back(proto_tensor);
    return &(tensor_list_.back());
  }
  void clear() override { tensor_list_.clear(); }

 private:
  proto::DistributedPredictReply *proto_reply_;
  std::vector<ProtoTensor> tensor_list_;
};

Status WorkerAgent::Run(const proto::DistributedPredictRequest &request, proto::DistributedPredictReply *reply) {
  if (session_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Model is not loaded";
  }
  Status status;
  try {
    MSI_TIME_STAMP_START(ExecuteModel)
    ProtoDistributedPredictRequest request_wrap(request);
    ProtoDistributedPredictReply reply_wrap(reply);
    status = session_->ExecuteModel(request_wrap, &reply_wrap, request.return_result(), request.subgraph());
    MSI_TIME_STAMP_END(ExecuteModel)
  } catch (const std::bad_alloc &ex) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: malloc memory failed";
  } catch (const std::runtime_error &ex) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: runtime error occurred: " << ex.what();
  } catch (const std::exception &ex) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: exception occurred: " << ex.what();
  } catch (...) {
    status = INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: exception occurred";
  }
  if (status != SUCCESS) {
    reply->Clear();
    auto error_msg = reply->mutable_error_msg();
    error_msg->set_error_code(status.StatusCode());
    error_msg->set_error_msg(status.StatusMessage());
  }
  return status;
}
}  // namespace serving
}  // namespace mindspore
