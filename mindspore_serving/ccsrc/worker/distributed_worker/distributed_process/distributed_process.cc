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
#include "worker/worker.h"
#include "common/proto_tensor.h"

namespace mindspore {
namespace serving {

grpc::Status MSDistributedImpl::AgentRegister(grpc::ServerContext *context, const proto::AgentRegisterRequest *request,
                                              proto::AgentRegisterReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  for (auto &spec : request->agent_spec()) {
    WorkerAgentSpec agent_spec;
    agent_spec.agent_address = request->address();
    GrpcTensorHelper::CopyFromAgentSpec(spec, &agent_spec);
    Status status(FAILED);
    status = servable_->RegisterAgent(agent_spec);
    if (status != SUCCESS) {
      MSI_LOG(ERROR) << "Agent Register FAILED";
    }
  }
  return grpc::Status::OK;
}

grpc::Status MSDistributedImpl::AgentExit(grpc::ServerContext *context, const proto::AgentExitRequest *request,
                                          proto::AgentExitReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  servable_->OnAgentExit();
  if (Worker::GetInstance().IsRunning()) {
    Worker::GetInstance().StopServable();
  }
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
}  // namespace serving
}  // namespace mindspore
