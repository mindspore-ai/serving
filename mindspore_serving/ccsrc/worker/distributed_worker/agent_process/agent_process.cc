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

#include "worker/distributed_worker/agent_process/agent_process.h"
#include "worker/distributed_worker/worker_agent.h"

namespace mindspore {
namespace serving {
grpc::Status MSAgentImpl::Exit(grpc::ServerContext *context, const proto::DistributedExitRequest *request,
                               proto::DistributedExitReply *reply) {
  MSI_LOG(INFO) << "Distributed Worker Exit";
  WorkerAgent::Instance().StopAgent(false);
  return grpc::Status::OK;
}

grpc::Status MSAgentImpl::Predict(grpc::ServerContext *context, const proto::DistributedPredictRequest *request,
                                  proto::DistributedPredictReply *reply) {
  MSI_LOG(INFO) << "Begin call service Eval";
  WorkerAgent::Instance().Run(*request, reply);
  MSI_LOG(INFO) << "End call service Eval";
  return grpc::Status::OK;
}
grpc::Status MSAgentImpl::Ping(grpc::ServerContext *context, const proto::PingRequest *request,
                               proto::PingReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  watcher_->RecvPing(request->address());
  return grpc::Status::OK;
}

grpc::Status MSAgentImpl::Pong(grpc::ServerContext *context, const proto::PongRequest *request,
                               proto::PongReply *reply) {
  MSI_EXCEPTION_IF_NULL(request);
  MSI_EXCEPTION_IF_NULL(reply);
  watcher_->RecvPong(request->address());
  return grpc::Status::OK;
}
}  // namespace serving
}  // namespace mindspore
