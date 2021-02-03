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

#include "worker/distributed_worker/distributed_servable.h"
#include <vector>
#include <string>
#include <set>
#include "worker/distributed_worker/notify_agent/notify_agent.h"
#include "common/exit_handle.h"

namespace mindspore {
namespace serving {

DistributedServable::~DistributedServable() { Clear(); }

std::string DistributedServable::GetServableName() const { return servable_name_; }

uint64_t DistributedServable::GetServableVersion() const { return version_number_; }

Status DistributedServable::Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output) {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return Status();
}
std::vector<TensorInfo> DistributedServable::GetInputInfos() const {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return input_infos_;
}

std::vector<TensorInfo> DistributedServable::GetOutputInfos() const {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return output_infos_;
}

uint64_t DistributedServable::GetBatchSize() const {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return batch_size_;
}

Status DistributedServable::GetDistributedServableConfig(DistributedServableConfig *config) const {
  *config = config_;
  return SUCCESS;
}

void DistributedServable::SetWaitAgentsPromise(bool flag) {
  if (!promise_set_flag_.test_and_set()) {
    agents_promise_.set_value(flag);
  }
}

Status DistributedServable::RegisterAgent(const WorkerAgentSpec &agent_spec) {
  std::unique_lock<std::mutex> lock{mutex_};

  if (agent_spec.rank_id < config_.distributed_meta.rank_size) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Invalid rank id " << agent_spec.rank_id << ", rank size " << config_.distributed_meta.rank_size;
  }
  DistributedAgentContext context;
  auto it = agent_spec_map_.find(agent_spec.rank_id);
  if (it != agent_spec_map_.end()) {
    MSI_LOG_WARNING << "rank_id " << agent_spec.rank_id << " has been registered";
    return SUCCESS;
  }
  context.agent_spec_ = agent_spec;
  std::shared_ptr<BaseNotifyAgent> notify_agent = std::make_shared<GrpcNotfiyAgent>(agent_spec.agent_address);
  context.notify_agent_ = notify_agent;
  agent_spec_map_[agent_spec.rank_id] = context;

  if (agent_spec_map_.size() >= config_.distributed_meta.rank_size) {
    SetWaitAgentsPromise(true);
  }
  return SUCCESS;
}

void DistributedServable::Clear() {
  std::unique_lock<std::mutex> lock{mutex_};
  for (auto &agent : agent_spec_map_) {
    agent.second.notify_agent_->Exit();
  }
  agent_spec_map_.clear();
  MSI_LOG_INFO << "End Clear servable";
}

Status DistributedServable::UnregisterAgent(const WorkerAgentSpec &agent_spec) {
  std::unique_lock<std::mutex> lock{mutex_};
  for (auto iter = agent_spec_map_.begin(); iter != agent_spec_map_.end();) {
    if (agent_spec.rank_id == iter->second.agent_spec_.rank_id) {
      iter = agent_spec_map_.erase(iter);
    } else {
      ++iter;
    }
  }
  SetWaitAgentsPromise(false);
  return SUCCESS;
}

Status DistributedServable::StartServable(const std::string &servable_directory, const std::string &servable_name,
                                          const std::string &rank_table_json_file, uint64_t version_number,
                                          uint64_t wait_agents_time_in_seconds) {
  if (model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has loaded";
  }
  version_number_ = version_number;
  servable_name_ = servable_name;
  rank_table_json_file_ = rank_table_json_file;
  ServableSignature signature;
  if (!ServableStorage::Instance().GetServableDef(servable_name, &signature)) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable '" << servable_name << "' has not been registered";
  }
  auto &meta = signature.servable_meta;
  if (meta.servable_type != kServableTypeDistributed) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Servable '" << servable_name << "' is not registered as distributed servable, " << meta.Repr();
  }
  config_.common_meta = meta.common_meta;
  config_.distributed_meta = meta.distributed_meta;

  auto status = InitConfigOnStartup(rank_table_json_file_);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Init with rank table on start up failed";
    return status;
  }
  status = CheckRankConfig();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check rank config failed";
    return status;
  }
  status = WaitAgentsReady(wait_agents_time_in_seconds);
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Waiting for ready of agents failed";
    return status;
  }
  status = CheckAgentsInfosAndInitTensorInfos();
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Check agents infos failed";
    return status;
  }
  model_loaded_ = true;
  return SUCCESS;
}

Status DistributedServable::InitConfigOnStartup(const std::string &rank_table_json_file) { return FAILED; }

Status DistributedServable::WaitAgentsReady(uint64_t wait_agents_time_in_seconds) {
  auto future = agents_promise_.get_future();
  if (wait_agents_time_in_seconds == 0) {
    wait_agents_time_in_seconds = UINT32_MAX;
  }
  const uint64_t kWaitMaxHundredMs = wait_agents_time_in_seconds * 10;
  uint64_t i;
  for (i = 0; i < kWaitMaxHundredMs; i++) {  //
    if (ExitSignalHandle::Instance().HasStopped()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Agents has stopped";
    }
    // waiting for 100ms
    if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      auto flag = future.get();
      if (!flag) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to starting all agents, maybe some error reported";
      }
      break;
    }
  }
  if (i >= kWaitMaxHundredMs) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Failed to wait for ready of all agents, current agents count: " << agent_spec_map_.size()
           << ", rank size: " << config_.distributed_meta.rank_size;
  }
  return SUCCESS;
}

Status DistributedServable::CompareTensorInfos(const std::vector<TensorInfo> &lefts,
                                               const std::vector<TensorInfo> &rights) {
  if (lefts.size() != rights.size()) {
    return INFER_STATUS(FAILED) << "Size not match, left: " << lefts.size() << ", right: " << rights.size();
  }
  auto tensor_info_as_str = [](const TensorInfo &tensor_info) {
    Status status = INFER_STATUS(SUCCESS) << "size: " << tensor_info.size << ", data type: " << tensor_info.data_type
                                          << ", shape: " << tensor_info.shape;
    return status.StatusMessage();
  };
  for (size_t k = 0; k < lefts.size(); k++) {
    auto &left = lefts[k];
    auto &right = rights[k];
    if (left.size != right.size || left.shape != right.shape || left.data_type != right.data_type) {
      return INFER_STATUS(FAILED) << "Index " << k << " tensor not match, left- " << tensor_info_as_str(left)
                                  << "; right- " << tensor_info_as_str(right);
    }
  }
  return SUCCESS;
}

Status DistributedServable::CheckAgentsInfosAndInitTensorInfos() {
  auto rank_size = config_.distributed_meta.rank_size;
  auto stage_size = config_.distributed_meta.stage_size;
  auto parallel_count = rank_size / stage_size;
  MSI_LOG_INFO << "Check agents infos, rank size :" << rank_size << ", stage size: " << stage_size
               << ", parallel count: " << parallel_count;
  if (agent_spec_map_.size() != rank_size) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Registered agents size " << agent_spec_map_.size() << " not match rank size " << rank_size;
  }

  input_infos_ = agent_spec_map_[0].agent_spec_.input_infos;
  output_infos_ = agent_spec_map_[rank_size - 1].agent_spec_.output_infos;
  batch_size_ = agent_spec_map_[0].agent_spec_.batch_size;
  if (input_infos_.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Rank " << 0 << " input count cannot be 0";
  }
  if (output_infos_.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Rank " << rank_size - 1 << " output count cannot be 0";
  }
  Status status;
  for (size_t i = 0; i < parallel_count; i++) {
    auto &agent_spec = agent_spec_map_[i];
    status = CompareTensorInfos(agent_spec.agent_spec_.input_infos, input_infos_);
    if (status != SUCCESS) {
      status = INFER_STATUS_LOG_ERROR(FAILED)
               << "Rank " << i << " input infos not match rank 0, details: " << status.StatusMessage();
      return status;
    }
  }
  for (size_t i = parallel_count; i < rank_size; i++) {
    auto &agent_spec = agent_spec_map_[i];
    if (!agent_spec.agent_spec_.input_infos.empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Expect rank " << i << " input count equal to 0";
    }
  }
  for (size_t i = 0; i < rank_size; i++) {
    auto &first_item = agent_spec_map_[i];
    for (size_t k = 0; k < parallel_count && i + k < rank_size; k++) {
      auto rank_id = i + k;
      auto &agent_spec = agent_spec_map_[i + k];
      status = CompareTensorInfos(agent_spec.agent_spec_.output_infos, first_item.agent_spec_.output_infos);
      if (status != SUCCESS) {
        status = INFER_STATUS_LOG_ERROR(FAILED) << "Rank " << rank_size << " output infos not match rank " << i
                                                << ", details: " << status.StatusMessage();
        return status;
      }
      if (agent_spec.agent_spec_.batch_size != 0 && agent_spec.agent_spec_.batch_size != batch_size_) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Expect rank " << rank_id << " batch size equal to 0 or rank 0 batch size " << batch_size_;
      }
    }
  }
  return SUCCESS;
}

Status DistributedServable::CheckRankConfig() {
  auto rank_size = config_.distributed_meta.rank_size;
  auto stage_size = config_.distributed_meta.stage_size;
  if (stage_size == 0 || rank_size == 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Rank size or stage size cannot be 0, rank size: " << rank_size << ", stage size: " << stage_size;
  }
  if (rank_size % stage_size != 0) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Rank size must be an integral multiple of stage size, rank size: " << rank_size
           << ", stage size: " << stage_size;
  }
  if (config_.rank_list.size() != rank_size) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Rank size " << config_.rank_list.size() << " declared in rank table file not equal to rank size "
           << rank_size << " declared in servable_config, rank json config file: " << rank_table_json_file_;
  }
  auto parallel_count = rank_size / stage_size;
  constexpr size_t card_count_per_machine = 8;
  if (stage_size == 1) {
    std::map<std::string, std::set<uint32_t>> device_map;
    for (size_t i = 0; i < rank_size; i++) {
      const auto &item = config_.rank_list[i];
      auto &device_id_list = device_map[item.ip];
      if (device_id_list.count(item.device_id) > 0) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Check rank table config failed, device id repeatedly used by rank "
                                              << i << " in device ip " << item.ip;
      }
      device_id_list.emplace(item.device_id);
    }
  } else {
    if (rank_size < card_count_per_machine) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Rank size " << rank_size << "must >= card count " << card_count_per_machine
             << " of one machine when stage size " << stage_size << " > 1";
    }
    if (parallel_count % card_count_per_machine != 0) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Parallel count " << parallel_count << " in one stage must be N * " << card_count_per_machine
             << "(card count of one machine), rank size: " << rank_size << ", stage size: " << stage_size;
    }
    for (size_t i = 0; i < rank_size; i += card_count_per_machine) {
      const auto &first_item = config_.rank_list[i];
      for (size_t k = 0; i + k < rank_size && k < card_count_per_machine; k++) {
        auto rank_id = i + k;
        const auto &item = config_.rank_list[rank_id];
        if (k != item.device_id) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Check rank table config failed, expected device id of rank " << rank_id << " to be " << k;
        }
        if (first_item.ip != item.ip) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Check rank table config failed, expected device ip " << item.ip << " of rank " << rank_id
                 << " to be equal with device ip " << first_item.ip << " of rank " << i;
        }
      }
    }
  }
  MSI_LOG_INFO << "Check rank table success, rank size: " << rank_size << ", stage size: " << stage_size
               << ", parallel count in one stage: " << parallel_count;
  return SUCCESS;
}

void DistributedServable::OnAgentFailed() { SetWaitAgentsPromise(false); }

}  // namespace serving
}  // namespace mindspore
