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
#include "worker/distributed_worker/notify_agent/notify_agent.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <thread>
#include "common/exit_handle.h"
#include "common/grpc_server.h"

namespace mindspore {
namespace serving {

GrpcNotfiyAgent::GrpcNotfiyAgent(const std::string &worker_address) {}

GrpcNotfiyAgent::~GrpcNotfiyAgent() = default;

Status GrpcNotfiyAgent::Exit() { return SUCCESS; }

Status GrpcNotfiyAgent::DispatchAsync(const proto::PredictRequest &request, proto::PredictReply *reply,
                                      DistributeCallback callback) {
  return SUCCESS;
}

}  // namespace serving
}  // namespace mindspore
