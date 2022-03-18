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

#include "worker/extra_worker/remote_call_model.h"
#include <unistd.h>
#include <memory>
#include "worker/notfiy_master/grpc_notify.h"
#include "common/proto_tensor.h"
#include "worker/worker.h"

namespace mindspore::serving {
Status RemoteCallModel::InitRemote(const std::string &servable_name, uint32_t version_number,
                                   const std::string &master_address,
                                   std::map<std::string, std::shared_ptr<ModelLoaderBase>> *models) {
  MSI_EXCEPTION_IF_NULL(models);
  proto::GetModelInfoReply reply;
  auto status = GrpcNotifyMaster::GetModelInfos(master_address, servable_name, version_number, &reply);
  if (status != SUCCESS) {
    return status;
  }
  if (reply.error_msg().error_code() != 0) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << reply.error_msg().error_msg();
  }
  std::map<std::string, ModelInfo> model_infos;
  GrpcTensorHelper::ConvertProtoModelInfos(reply.model_infos(), &model_infos);

  for (auto &model_it : model_infos) {
    auto &model_name = model_it.first;
    auto &model_info = model_it.second;
    auto model_loader = std::make_shared<RemoteCallModel>();
    (void)models->emplace(model_name, model_loader);
    status = model_loader->InitModel(model_name, version_number, model_info);
    if (status != SUCCESS) {
      for (auto &item : *models) {
        item.second->Clear();
      }
      return status;
    }
  }
  return SUCCESS;
}

Status RemoteCallModel::InitModel(const std::string &model_key, uint32_t version_number, const ModelInfo &model_info) {
  model_key_ = model_key;
  batch_size_ = model_info.batch_size;
  if (batch_size_ == 0) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Batch size cannot be 0";
  }
  auto &subgraph_infos = model_info.sub_graph_infos;
  subgraph_contexts_.resize(subgraph_infos.size());
  for (size_t i = 0; i < subgraph_infos.size(); i++) {
    auto &subgraph_info = subgraph_infos[i];
    RemoteCallModelContext &context = subgraph_contexts_[i];
    context.model_name = model_key;
    context.version_number = version_number;
    context.subgraph = i;
    context.input_infos = subgraph_info.input_infos;
    for (auto &tensor_info : subgraph_info.output_infos) {
      TensorInfoOutput output_info;
      output_info.tensor_info = tensor_info;
      context.output_infos.push_back(output_info);
    }
  }
  auto status = InitModelExecuteInfo();
  if (status != SUCCESS) {
    return status;
  }
  return SUCCESS;
}

std::vector<TensorInfo> RemoteCallModel::GetInputInfos(uint64_t subgraph) const {
  if (subgraph >= subgraph_contexts_.size()) {
    MSI_LOG_EXCEPTION << "Cannot find subgraph " << subgraph << " in model " << model_key_;
  }
  return subgraph_contexts_[subgraph].input_infos;
}

std::vector<TensorInfo> RemoteCallModel::GetOutputInfos(uint64_t subgraph) const {
  if (subgraph >= subgraph_contexts_.size()) {
    MSI_LOG_EXCEPTION << "Cannot find subgraph " << subgraph << " in model " << model_key_;
  }
  std::vector<TensorInfo> output_tensors;
  for (auto &item : subgraph_contexts_[subgraph].output_infos) {
    // cppcheck-suppress useStlAlgorithm
    output_tensors.push_back(item.tensor_info);
  }
  return output_tensors;
}

uint64_t RemoteCallModel::GetBatchSize() const { return batch_size_; }

uint64_t RemoteCallModel::GetGraphNum() const { return subgraph_contexts_.size(); }

void RemoteCallModel::Clear() { subgraph_contexts_.clear(); }

Status RemoteCallModel::Predict(const std::vector<InstanceData> &inputs, std::vector<ResultInstance> *outputs,
                                uint64_t subgraph) {
  auto notify_master = Worker::GetInstance().GetGrpcNotifyMaster();
  if (notify_master == nullptr) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Get notify master failed";
  }
  if (subgraph >= subgraph_contexts_.size()) {
    MSI_LOG_EXCEPTION << "Cannot find subgraph " << subgraph << " in model " << model_key_;
  }
  return notify_master->CallModel(subgraph_contexts_[subgraph], inputs, outputs);
}

Status RemoteCallModel::InitModelExecuteInfo() {
  auto pid = getpid();
  Status status;
  constexpr uint32_t cache_times = 3;
  auto &shared_memory = SharedMemoryAllocator::Instance();
  for (auto &subgraph : subgraph_contexts_) {
    for (size_t i = 0; i < subgraph.input_infos.size(); i++) {
      auto &tensor_info = subgraph.input_infos[i];
      uint64_t size_one_batch = tensor_info.size;
      if (!tensor_info.is_no_batch_dim) {
        size_one_batch = size_one_batch / batch_size_;
      }
      auto memory_key = model_key_ + "_subgraph" + std::to_string(subgraph.subgraph) + "_input" + std::to_string(i) +
                        "_pid" + std::to_string(pid);
      uint64_t init_count = batch_size_ * cache_times;
      status = shared_memory.NewMemoryBuffer(memory_key, size_one_batch, init_count);
      if (status != SUCCESS) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Init input shared memory failed, item size: " << size_one_batch << ", initial count: " << init_count;
      }
      subgraph.request_memory.push_back(memory_key);
    }
    for (size_t i = 0; i < subgraph.output_infos.size(); i++) {
      auto &output_info = subgraph.output_infos[i];
      auto &tensor_info = output_info.tensor_info;
      if (tensor_info.is_no_batch_dim) {
        output_info.shape_one_batch = tensor_info.shape;
        output_info.size_one_batch = tensor_info.size;
      } else {
        output_info.shape_one_batch = tensor_info.shape;
        (void)output_info.shape_one_batch.erase(output_info.shape_one_batch.begin());
        // the batch size has been checked in WorkerExecutor
        output_info.size_one_batch = tensor_info.size / batch_size_;
      }
      auto memory_key = model_key_ + "_subgraph" + std::to_string(subgraph.subgraph) + "_output" + std::to_string(i) +
                        "_pid" + std::to_string(pid);
      uint64_t init_count = batch_size_ * cache_times;
      status = shared_memory.NewMemoryBuffer(memory_key, output_info.size_one_batch, init_count);
      if (status != SUCCESS) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Init output shared memory failed, item size: " << output_info.size_one_batch
               << ", initial count: " << init_count;
      }
      subgraph.reply_memory.push_back(memory_key);
    }
  }
  return SUCCESS;
}
}  // namespace mindspore::serving
