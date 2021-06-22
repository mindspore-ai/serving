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

#include "worker/distributed_worker/distributed_process/distributed_process.h"
#include <vector>
#include "worker/worker.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {

grpc::Status MSDistributedImpl::AgentRegister(grpc::ServerContext *context, const proto::AgentRegisterRequest *request,
                                              proto::AgentRegisterReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  std::vector<WorkerAgentSpec> agent_specs;
  for (auto &spec : request->agent_spec()) {
    WorkerAgentSpec agent_spec;
    agent_spec.agent_address = request->address();
    GrpcTensorHelper::CopyFromAgentSpec(spec, &agent_spec);
    agent_specs.push_back(agent_spec);
  }
  if (agent_specs.size() == 0) {
    MSI_LOG(ERROR) << "Agent Register FAILED, agent_specs size is 0";
  }
  Status status(FAILED);
  status = servable_->RegisterAgent(agent_specs);
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "Agent Register FAILED";
  }
  watcher_->StartWatch(request->address());
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::AgentExit(grpc::ServerContext *context, const proto::AgentExitRequest *request,
                                          proto::AgentExitReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  if (request->address_choice_case() == proto::AgentExitRequest::kAddress) {
    watcher_->StopWatch(request->address());
  }
  MSI_LOG_INFO << "Agent exit, address: '" << request->address() << "', agent ip: '" << request->agent_ip() << "'";
  servable_->OnAgentExit();
  Worker::GetInstance().StopServable();
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::AgentFailed(grpc::ServerContext *context, const proto::AgentFailedRequest *request,
                                            proto::AgentFailedReply *reply) {
  if (Worker::GetInstance().IsRunning()) {
    MSI_LOG_ERROR << "Expect worker should not be running";
    Worker::GetInstance().StopServable();
  } else {
    servable_->OnAgentFailed();
  }
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::AgentConfigAcquire(grpc::ServerContext *context,
                                                   const proto::AgentConfigAcquireRequest *request,
                                                   proto::AgentConfigAcquireReply *reply) {
  Status status(FAILED);
  DistributedServableConfig agent_config;
  status = servable_->GetDistributedServableConfig(&agent_config);
  if (status != SUCCESS) {
    MSI_LOG(ERROR) << "Get distributed servable config failed";
    return grpc::Status::CANCELLED;
  }

  MSI_LOG(INFO) << "Begin to set DistributedServableConfig info in reply message";
  // set reply message:AgentConfigAcquireReply, parameter:rank_table_content
  reply->set_rank_table_content(agent_config.rank_table_content);
  // set reply message:AgentConfigAcquireReply, parameter:rank_list
  auto &agent_rank_list = agent_config.rank_list;
  for (auto &agent_rank : agent_rank_list) {
    auto rank_list = reply->add_rank_list();
    rank_list->set_ip(agent_rank.ip);
    rank_list->set_device_id(agent_rank.device_id);
  }
  // set reply message:AgentConfigAcquireReply, parameter:common_meta
  auto reply_common_meta = reply->mutable_common_meta();
  reply_common_meta->set_servable_name(agent_config.common_meta.servable_name);
  reply_common_meta->set_with_batch_dim(agent_config.common_meta.with_batch_dim);
  auto &without_batch_dim_inputs_list = agent_config.common_meta.without_batch_dim_inputs;
  for (auto &without_batch_dim_input : without_batch_dim_inputs_list) {
    reply_common_meta->add_without_batch_dim_inputs(without_batch_dim_input);
  }
  reply_common_meta->set_inputs_count(agent_config.common_meta.inputs_count);
  reply_common_meta->set_outputs_count(agent_config.common_meta.outputs_count);

  // set reply message:AgentConfigAcquireReply, parameter:distributed_meta
  auto reply_distributed_meta = reply->mutable_distributed_meta();
  reply_distributed_meta->set_rank_size(agent_config.distributed_meta.rank_size);
  reply_distributed_meta->set_stage_size(agent_config.distributed_meta.stage_size);
  MSI_LOG(INFO) << "Success to set DistributedServableConfig info in reply message";

  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::Ping(grpc::ServerContext *context, const proto::PingRequest *request,
                                     proto::PingReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  watcher_->RecvPing(request->address());
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::Pong(grpc::ServerContext *context, const proto::PongRequest *request,
                                     proto::PongReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  watcher_->RecvPong(request->address());
  return grpc::Status::OK;
}

}  // namespace serving
}  // namespace mindspore
