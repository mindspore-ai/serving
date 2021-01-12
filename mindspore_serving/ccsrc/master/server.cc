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

#include "master/server.h"

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/proto_tensor.h"
#include "common/serving_common.h"
#include "common/exit_handle.h"
#include "master/dispacther.h"
#include "master/grpc/grpc_process.h"
#include "master/restful/http_process.h"
#include "worker/context.h"
#include "master/grpc/grpc_server.h"

namespace mindspore {
namespace serving {

Status Server::StartGrpcServer(const std::string &ip, uint32_t grpc_port, int max_msg_mb_size) {
  ExitSignalHandle::Instance().Start();  // handle ctrl+c to exit
  if (max_msg_mb_size > gRpcMaxMBMsgSize) {
    MSI_LOG_WARNING << "The maximum Serving gRPC message size is 512MB and will be updated from " << max_msg_mb_size
                    << "MB to 512MB";
    max_msg_mb_size = gRpcMaxMBMsgSize;
  }
  grpc_async_server_ = std::make_unique<MSServiceServer>(std::make_shared<MSServiceImpl>(dispatcher_), ip, grpc_port);
  return grpc_async_server_->Init();
}

Status Server::StartGrpcMasterServer(const std::string &ip, uint32_t grpc_port) {
  ExitSignalHandle::Instance().Start();  // handle ctrl+c to exit
  return grpc_manager_server_.Start(std::make_shared<MSMasterImpl>(dispatcher_), ip, grpc_port, gRpcMaxMBMsgSize,
                                    "Master");
}

Status Server::StartRestfulServer(const std::string &ip, uint32_t restful_port, int max_msg_mb_size,
                                  int time_out_second) {
  ExitSignalHandle::Instance().Start();  // handle ctrl+c to exit
  return restful_server_.Start(ip, restful_port, max_msg_mb_size, time_out_second);
}

void Server::Clear() {
  dispatcher_->Clear();
  restful_server_.Stop();
  grpc_server_.Stop();
  grpc_manager_server_.Stop();
}

Server::Server() = default;

Server &Server::Instance() {
  static Server server;
  return server;
}

Server::~Server() = default;

}  // namespace serving
}  // namespace mindspore
