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

#include "worker/distributed_worker/distributed_model_loader.h"
#include <vector>
#include <string>
#include <set>
#include "worker/distributed_worker/notify_agent/notify_agent.h"
#include "worker/worker.h"
#include "common/exit_handle.h"
#include "common/proto_tensor.h"
#include "worker/servable_register.h"

namespace mindspore {
namespace serving {
struct DistributedPredictMsg {
  proto::DistributedPredictReply reply;
  std::promise<void> promise = std::promise<void>();
  Status status = FAILED;
  std::future<void> future = promise.get_future();
};

DistributedModelLoader::~DistributedModelLoader() { Clear(); }

uint64_t DistributedModelLoader::GetGraphNum() const { return graph_num_; }

Status DistributedModelLoader::Predict(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                                       uint64_t subgraph) {
  Status status(SUCCESS);
  if (config_.distributed_meta.enable_pipeline_infer) {
    std::shared_lock<std::shared_mutex> lock{rw_mutex_};
    status = PredictInner(input, output, subgraph);
  } else {
    std::unique_lock<std::shared_mutex> lock{rw_mutex_};
    status = PredictInner(input, output, subgraph);
  }
  if (status != SUCCESS) {
    MSI_LOG_ERROR << "Predict error happened, now exit distributed servable";
    Worker::GetInstance().StopServable();
  }
  return status;
}

Status DistributedModelLoader::PredictInner(const std::vector<TensorBasePtr> &input, std::vector<TensorBasePtr> *output,
                                            uint64_t subgraph) {
  MSI_EXCEPTION_IF_NULL(output);
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }

  proto::DistributedPredictRequest request;
  proto::DistributedPredictRequest empty_request;
  request.set_subgraph(subgraph);
  request.set_subgraph(subgraph);
  for (const auto &tensor_ptr : input) {
    auto tensor = request.add_inputs();
    ProtoTensor proto_tensor(tensor);
    proto_tensor.assign(*tensor_ptr);
  }

  auto rank_size = config_.distributed_meta.rank_size;
  auto stage_size = config_.distributed_meta.stage_size;
  if (rank_size != agent_spec_map_.size()) {
    MSI_LOG_EXCEPTION << "agent_spec_map_ size " << agent_spec_map_.size() << " not match rank size " << rank_size;
  }
  auto agent_num_per_stage = rank_size / stage_size;
  auto result_agent_id = rank_size - 1;

  auto msg_list = std::make_shared<std::vector<DistributedPredictMsg>>(rank_size);
  request.set_return_result(false);
  empty_request.set_return_result(false);

  std::unique_lock<std::mutex> wait_lock(wait_mutex_);
  for (size_t i = 0; i < rank_size; ++i) {
    AsyncPredictCallback callback = [msg_list, i](const Status &status) {
      msg_list->at(i).status = status;
      msg_list->at(i).promise.set_value();
    };
    if (i < agent_num_per_stage || all_stage_has_input_) {
      if (i == result_agent_id) {
        request.set_return_result(true);
      }
      agent_spec_map_[i].notify_agent_->DispatchAsync(request, &msg_list->at(i).reply, callback);
    } else {
      if (i == result_agent_id) {
        empty_request.set_return_result(true);
      }
      agent_spec_map_[i].notify_agent_->DispatchAsync(empty_request, &msg_list->at(i).reply, callback);
    }
  }
  wait_lock.unlock();

  for (size_t rank_id = 0; rank_id < msg_list->size(); ++rank_id) {
    auto &predict_msg = msg_list->at(rank_id);
    auto &future = predict_msg.future;
    const uint64_t kWaitMaxHundredMs = 10 * 10;  // waiting for 10s
    uint64_t k;
    for (k = 0; k < kWaitMaxHundredMs; k++) {
      if (ExitSignalHandle::Instance().HasStopped()) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Worker has stopped";
      }
      // waiting for 100ms
      if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
        break;
      }
    }
    if (k >= kWaitMaxHundredMs) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Failed to wait for result of rank " << rank_id;
    }
    auto status = predict_msg.status;
    if (status != SUCCESS) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Error happened on get result of rank " << rank_id << ": " << status.StatusMessage();
    }
    auto &reply = predict_msg.reply;
    if (reply.has_error_msg() && reply.error_msg().error_code() != 0) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Error happened on get result of rank " << rank_id << ": " << reply.error_msg().error_msg();
    }
  }

  auto &reply = msg_list->at(result_agent_id).reply;
  for (int i = 0; i < reply.outputs_size(); ++i) {
    auto p = std::make_shared<ProtoTensor>(reply.mutable_outputs(i));
    auto tensor_ptr = std::make_shared<Tensor>(p->data_type(), p->shape(), p->data(), p->data_size());
    output->push_back(tensor_ptr);
  }
  return SUCCESS;
}

std::vector<TensorInfo> DistributedModelLoader::GetInputInfos(uint64_t subgraph) const {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  auto iter = input_infos_.find(subgraph);
  if (iter != input_infos_.end()) {
    return iter->second;
  }
  MSI_LOG_EXCEPTION << "subgraph: " << subgraph << " is not existed";
  return {};
}

std::vector<TensorInfo> DistributedModelLoader::GetOutputInfos(uint64_t subgraph) const {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  auto iter = output_infos_.find(subgraph);
  if (iter != output_infos_.end()) {
    return iter->second;
  }
  MSI_LOG_EXCEPTION << "subgraph: " << subgraph << " is not existed";
  return {};
}

uint64_t DistributedModelLoader::GetBatchSize() const {
  if (!model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has not been loaded";
  }
  return batch_size_;
}

Status DistributedModelLoader::GetDistributedServableConfig(DistributedServableConfig *config) const {
  if (!config_loaded_) {
    return INFER_STATUS(FAILED) << "Config not loaded";
  }
  *config = config_;
  return SUCCESS;
}

void DistributedModelLoader::SetWaitAgentsPromise(bool flag) {
  if (!promise_set_flag_.test_and_set()) {
    agents_promise_.set_value(flag);
    registered_end_flag_ = true;
  }
}

Status DistributedModelLoader::RegisterAgent(const std::vector<WorkerAgentSpec> &agent_specs) {
  std::unique_lock<std::shared_mutex> lock{rw_mutex_};
  if (registered_end_flag_) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Distributed servable has ended up registration";
  }
  if (agent_specs.empty()) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "The number of graph cannot be 0";
  }
  if (agent_specs[0].rank_id >= config_.distributed_meta.rank_size) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Invalid rank id " << agent_specs[0].rank_id << ", rank size " << config_.distributed_meta.rank_size;
  }
  DistributedAgentContext context;
  auto it = agent_spec_map_.find(agent_specs[0].rank_id);
  if (it != agent_spec_map_.end()) {
    MSI_LOG_WARNING << "rank_id " << agent_specs[0].rank_id << " has been registered";
    return SUCCESS;
  }
  context.agent_spec_ = agent_specs;
  std::shared_ptr<BaseNotifyAgent> notify_agent = std::make_shared<GrpcNotifyAgent>(agent_specs[0].agent_address);
  context.notify_agent_ = notify_agent;
  agent_spec_map_[agent_specs[0].rank_id] = context;
  MSI_LOG_INFO << "Rank " << agent_specs[0].rank_id << " been registered";

  if (agent_spec_map_.size() >= config_.distributed_meta.rank_size) {
    SetWaitAgentsPromise(true);
  }
  return SUCCESS;
}

void DistributedModelLoader::Clear() {
  std::unique_lock<std::shared_mutex> lock{rw_mutex_};
  for (auto &agent : agent_spec_map_) {
    agent.second.notify_agent_->Exit();
  }
  agent_spec_map_.clear();
  model_loaded_ = false;
  MSI_LOG_INFO << "End clear distributed servable";
}

Status DistributedModelLoader::OnAgentExit() {
  std::unique_lock<std::shared_mutex> lock{rw_mutex_};
  MSI_LOG_INFO << "Worker agent notify exit";
  SetWaitAgentsPromise(false);
  model_loaded_ = false;
  return SUCCESS;
}

Status DistributedModelLoader::LoadModel(const std::string &servable_name, const std::string &rank_table_json_file,
                                         uint64_t wait_agents_time_in_seconds) {
  if (model_loaded_) {
    MSI_LOG_EXCEPTION << "Model has loaded";
  }
  rank_table_json_file_ = rank_table_json_file;
  const ServableSignature &signature = ServableRegister::Instance().GetServableSignature();
  if (signature.servable_name != servable_name) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Servable '" << servable_name << "' has not been registered";
  }
  if (signature.model_metas.size() != 1) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Distributed servable '" << servable_name
                                          << "' has not been declared or has been declared more than once, "
                                          << "declared number: " << signature.model_metas.size();
  }
  if (signature.servable_type != kServableTypeDistributed) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Servable '" << servable_name << "' is not registered as distributed servable";
  }
  auto &meta = signature.model_metas[0];
  model_key_ = meta.common_meta.model_key;
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
  config_loaded_ = true;
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

std::string RealPath(const std::string &path) {
  // Return absolute path when path is accessible
  std::string res;
  char resolved_path[PATH_MAX] = {0};
  if (realpath(path.c_str(), resolved_path) != nullptr) {
    res = resolved_path;
  }
  return res;
}

Status DistributedModelLoader::InitConfigOnStartup(const std::string &rank_table_json_file) {
  std::string rank_table_json_abs_path = RealPath(rank_table_json_file);
  if (rank_table_json_abs_path.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "failed to get realpath of：" << rank_table_json_file.c_str();
  }

  MSI_LOG(INFO) << "Begin to parser rank table json file: " << rank_table_json_file.c_str();
  std::ifstream json_file(rank_table_json_abs_path);
  if (!json_file.is_open()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "failed to open rank table file：" << rank_table_json_file.c_str();
  }
  std::stringstream buffer;
  buffer << json_file.rdbuf();
  config_.rank_table_content = buffer.str();

  json rank_table_json;
  try {
    rank_table_json = nlohmann::json::parse(config_.rank_table_content);
  } catch (json::parse_error &e) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "parse error:" << e.what();
  } catch (json::out_of_range &e) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "out of range:" << e.what();
  } catch (json::exception &e) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "Json exception:" << e.what();
  }

  if (!rank_table_json.is_object()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << rank_table_json_file.c_str() << " is not json object";
  }

  if (rank_table_json.find("group_list") != rank_table_json.end()) {
    return ParserRankTableWithGroupList(rank_table_json_file, rank_table_json);
  } else {
    return ParserRankTableWithServerList(rank_table_json_file, rank_table_json);
  }
}

json DistributedModelLoader::ParserArrayInJson(const json &json_array, const std::string &str) {
  json temp_array;
  auto iter = json_array.find(str);
  if (iter == json_array.end()) {
    MSI_LOG_ERROR << "Check rank table file failed" << str << "in file is not find";
    return temp_array;
  }
  if (!iter->is_array()) {
    MSI_LOG_ERROR << "Check rank table file failed" << str << "in file is not array";
    return temp_array;
  }
  temp_array = json_array.at(str);
  return temp_array;
}

std::string DistributedModelLoader::ParserStringInJson(const json &json_str, const std::string &str) {
  std::string temp_str;
  auto iter = json_str.find(str);
  if (iter == json_str.end()) {
    MSI_LOG_ERROR << "Check rank table file failed" << str << "in file is not find";
    return temp_str;
  }
  if (!iter->is_string()) {
    MSI_LOG_ERROR << "Check rank table file failed" << str << "in file is not string";
    return temp_str;
  }
  json temp_json_str = json_str.at(str);
  temp_str = temp_json_str.get<std::string>();
  return temp_str;
}

Status DistributedModelLoader::ParserRankTableWithGroupList(const std::string &rank_table_json_file,
                                                            const json &rank_table_json) {
  MSI_LOG_INFO << "Begin to parser rank table with group list";
  auto server_list = ParserArrayInJson(rank_table_json, "group_list");
  if (server_list.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "group_list attr is empty in" << rank_table_json_file.c_str();
  }

  size_t rank_id = 0;
  for (auto &server : server_list) {
    auto instance_list = ParserArrayInJson(server, "instance_list");
    if (instance_list.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "instance_list attr is empty in" << rank_table_json_file.c_str();
    }

    for (auto &instance : instance_list) {
      auto str_server_id = ParserStringInJson(instance, "server_id");
      if (str_server_id.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "server_id attr is empty in" << rank_table_json_file.c_str();
      }

      OneRankConfig one_rank_config;
      one_rank_config.ip = str_server_id;

      auto devices = ParserArrayInJson(instance, "devices");
      if (devices.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "devices attr is empty in" << rank_table_json_file.c_str();
      }

      auto str_device_id = ParserStringInJson(devices.at(0), "device_id");
      if (str_device_id.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "device_id attr is empty in" << rank_table_json_file.c_str();
      }
      uint32_t temp_device_id;
      auto status = ConvertStr2Int(rank_table_json_file, str_device_id, "device_id", &temp_device_id);
      if (status != SUCCESS) {
        MSI_LOG_ERROR << "Convert device_id from string to int failed";
        return status;
      }

      auto str_rank_id = ParserStringInJson(instance, "rank_id");
      if (str_rank_id.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "rank_id attr is empty in" << rank_table_json_file.c_str();
      }
      uint32_t temp_rank_id;
      status = ConvertStr2Int(rank_table_json_file, str_rank_id, "rank_id", &temp_rank_id);
      if (status != SUCCESS) {
        MSI_LOG_ERROR << "Convert rank_id from string to int failed";
        return status;
      }

      if (rank_id != temp_rank_id) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "device size not match rank_id in" << rank_table_json_file.c_str();
      }
      rank_id++;
      one_rank_config.device_id = temp_device_id;
      config_.rank_list.push_back(one_rank_config);
    }
  }
  MSI_LOG(INFO) << "Success parser rank table json file with group list and save to DistributedServableConfig";

  return SUCCESS;
}
Status DistributedModelLoader::ConvertStr2Int(const std::string &rank_table_json_file, const std::string &para_str,
                                              const std::string &para_key, uint32_t *para_int) const {
  uint32_t parsed_value = 0;
  constexpr uint32_t decimal_times = 10;
  for (auto c : para_str) {
    if (c < '0' || c > '9') {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
             << para_key << "attr is invalid argument in" << rank_table_json_file.c_str();
    }
    parsed_value = parsed_value * decimal_times + c - '0';
  }
  if (std::to_string(parsed_value) != para_str) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
           << para_key << "attr is invalid argument in" << rank_table_json_file.c_str();
  }
  *para_int = parsed_value;
  return SUCCESS;
}

Status DistributedModelLoader::ParserRankTableWithServerList(const std::string &rank_table_json_file,
                                                             const json &rank_table_json) {
  MSI_LOG_INFO << "Begin to parser rank table with server list";
  auto server_list = ParserArrayInJson(rank_table_json, "server_list");
  if (server_list.empty()) {
    return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "server_list attr is empty in" << rank_table_json_file.c_str();
  }

  size_t rank_id = 0;
  for (auto &server : server_list) {
    auto server_id = ParserStringInJson(server, "server_id");
    if (server_id.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "server_id attr is empty in" << rank_table_json_file.c_str();
    }

    auto device_list = ParserArrayInJson(server, "device");
    if (device_list.empty()) {
      return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "device attr is empty in" << rank_table_json_file.c_str();
    }

    for (auto &device : device_list) {
      OneRankConfig one_rank_config;
      one_rank_config.ip = server_id;
      auto str_device_id = ParserStringInJson(device, "device_id");
      if (str_device_id.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "device_id attr is empty in" << rank_table_json_file.c_str();
      }
      uint32_t temp_device_id;
      auto status = ConvertStr2Int(rank_table_json_file, str_device_id, "device_id", &temp_device_id);
      if (status != SUCCESS) {
        MSI_LOG_ERROR << "Convert device_id from string to int failed";
        return status;
      }

      auto str_rank_id = ParserStringInJson(device, "rank_id");
      if (str_rank_id.empty()) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS) << "rank_id attr is empty in" << rank_table_json_file.c_str();
      }
      uint32_t temp_rank_id;
      status = ConvertStr2Int(rank_table_json_file, str_rank_id, "rank_id", &temp_rank_id);
      if (status != SUCCESS) {
        MSI_LOG_ERROR << "Convert rank_id from string to int failed";
        return status;
      }

      if (rank_id != temp_rank_id) {
        return INFER_STATUS_LOG_ERROR(INVALID_INPUTS)
               << "device size not match rank_id in" << rank_table_json_file.c_str();
      }
      rank_id++;
      one_rank_config.device_id = temp_device_id;
      config_.rank_list.push_back(one_rank_config);
    }
  }
  MSI_LOG(INFO) << "Success parser rank table json file with server list and save to DistributedServableConfig";

  return SUCCESS;
}

Status DistributedModelLoader::WaitAgentsReady(uint64_t wait_agents_time_in_seconds) {
  MSI_LOG_INFO << "Begin waiting ready of all agents";
  auto future = agents_promise_.get_future();
  if (wait_agents_time_in_seconds == 0) {
    wait_agents_time_in_seconds = UINT32_MAX;
  }
  const uint64_t kWaitMaxHundredMs = wait_agents_time_in_seconds * 10;
  uint64_t i;
  for (i = 0; i < kWaitMaxHundredMs; i++) {  //
    if (ExitSignalHandle::Instance().HasStopped()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Worker or Agents has stopped";
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
  MSI_LOG_INFO << "Success waiting ready of all agents";
  return SUCCESS;
}

Status DistributedModelLoader::CompareTensorInfos(const std::vector<TensorInfo> &lefts,
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

Status DistributedModelLoader::CheckAgentsInfosAndInitTensorInfos() {
  auto rank_size = config_.distributed_meta.rank_size;
  auto stage_size = config_.distributed_meta.stage_size;
  auto parallel_count = rank_size / stage_size;
  MSI_LOG_INFO << "Check agents infos, rank size :" << rank_size << ", stage size: " << stage_size
               << ", parallel count(rank size/stage size): " << parallel_count;
  if (agent_spec_map_.size() != rank_size) {
    return INFER_STATUS_LOG_ERROR(FAILED)
           << "Registered agents size " << agent_spec_map_.size() << " not match rank size " << rank_size;
  }
  graph_num_ = agent_spec_map_[0].agent_spec_.size();
  for (size_t i = 1; i < rank_size; i++) {
    if (graph_num_ != agent_spec_map_[i].agent_spec_.size()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "The number of graph not match in different agent";
    }
  }
  batch_size_ = agent_spec_map_[0].agent_spec_[0].batch_size;
  for (size_t subgraph = 0; subgraph < agent_spec_map_[0].agent_spec_.size(); subgraph++) {
    input_infos_[subgraph] = agent_spec_map_[0].agent_spec_[subgraph].input_infos;
    output_infos_[subgraph] = agent_spec_map_[rank_size - 1].agent_spec_[subgraph].output_infos;
    if (input_infos_[subgraph].empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Rank " << 0 << " input count cannot be 0";
    }
    if (output_infos_[subgraph].empty()) {
      return INFER_STATUS_LOG_ERROR(FAILED) << "Rank " << rank_size - 1 << " output count cannot be 0";
    }
    const auto &input_infos = input_infos_[subgraph];
    Status status;
    for (size_t i = 0; i < parallel_count; i++) {
      auto &agent_spec = agent_spec_map_[i].agent_spec_[subgraph];
      status = CompareTensorInfos(agent_spec.input_infos, input_infos);
      if (status != SUCCESS) {
        status =
          INFER_STATUS_LOG_ERROR(FAILED)
          << "Rank " << i << " input infos not match rank 0, subgraph: " << subgraph
          << ", you can check if the actual stage size of the distributed model matches the stage size declared in "
             "servable_config.py, details: "
          << status.StatusMessage();
        return status;
      }
    }
    for (size_t i = parallel_count; i < rank_size; i++) {
      auto &agent_spec = agent_spec_map_[i].agent_spec_[subgraph];
      if (agent_spec.input_infos.empty()) {
        if (all_stage_has_input_) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Expect stage 0(other stages have empty inputs) or all stages have same inputs, detect rank "
                 << (i - 1) << " input count is " << agent_spec.input_infos.size() << ", but rank " << i
                 << " input count is 0, subgraph: " << subgraph;
        }
        continue;
      }
      status = CompareTensorInfos(agent_spec.input_infos, input_infos);
      if (status != SUCCESS) {
        return INFER_STATUS_LOG_ERROR(FAILED)
               << "Expect stage 0(other stages have empty inputs) or all stages have same inputs, detect rank " << i - 1
               << " and rank " << i << " inputs are different, subgraph: " << subgraph
               << ", details: " << status.StatusMessage();
      }
      all_stage_has_input_ = true;
    }
    for (size_t i = 0; i < rank_size; i += parallel_count) {
      const auto &first_item = agent_spec_map_[i].agent_spec_[subgraph];
      for (size_t k = 0; k < parallel_count && i + k < rank_size; k++) {
        auto rank_id = i + k;
        const auto &agent_spec = agent_spec_map_[i + k].agent_spec_[subgraph];
        status = CompareTensorInfos(agent_spec.output_infos, first_item.output_infos);
        if (status != SUCCESS) {
          status = INFER_STATUS_LOG_ERROR(FAILED)
                   << "Rank " << rank_id << " output infos not match rank " << i << ", subgraph: " << subgraph
                   << ", details: " << status.StatusMessage();
          return status;
        }
        if (agent_spec.batch_size != 0 && agent_spec.batch_size != batch_size_) {
          return INFER_STATUS_LOG_ERROR(FAILED)
                 << "Expect rank " << rank_id << " batch size  " << agent_spec.batch_size
                 << " equal to 0 or rank 0's batch size " << batch_size_ << ", subgraph: " << subgraph;
        }
      }
    }
  }
  return SUCCESS;
}

Status DistributedModelLoader::CheckRankConfig() {
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
           << "Rank size " << config_.rank_list.size() << " declared in rank table file '" << rank_table_json_file_
           << "' not equal to " << rank_size << " declared in servable config";
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
      if (item.device_id >= card_count_per_machine) {
        return INFER_STATUS_LOG_ERROR(FAILED) << "Check rank table config failed, device id cannot larger than 8";
      }
      (void)device_id_list.emplace(item.device_id);
    }
  } else {
    if (rank_size < card_count_per_machine) {
      return INFER_STATUS_LOG_ERROR(FAILED)
             << "Rank size " << rank_size << "must >= card count " << card_count_per_machine
             << " of one machine when stage size " << stage_size << " > 1";
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

void DistributedModelLoader::OnAgentFailed() {
  MSI_LOG_INFO << "Worker agent notify failed";
  SetWaitAgentsPromise(false);
}
}  // namespace serving
}  // namespace mindspore
