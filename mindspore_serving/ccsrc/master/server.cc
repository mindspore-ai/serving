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

#include <memory>
#include <string>

#include "common/serving_common.h"
#include "master/grpc/grpc_process.h"
#include "master/grpc/grpc_server.h"

namespace mindspore {
namespace serving {
Status Server::StartGrpcServer(const std::string &socket_address, const SSLConfig &ssl_config, int max_msg_mb_size) {
  if (grpc_async_server_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: Serving gRPC server is already running";
  }
  if (max_msg_mb_size > gRpcMaxMBMsgSize) {
    MSI_LOG_WARNING << "The maximum Serving gRPC message size is 512MB and will be updated from " << max_msg_mb_size
                    << "MB to 512MB";
    max_msg_mb_size = gRpcMaxMBMsgSize;
  }
  grpc_async_server_ = std::make_shared<ServiceGrpcServer>(dispatcher_);
  return grpc_async_server_->Start(socket_address, ssl_config, max_msg_mb_size, "Serving gRPC");
}

Status Server::StartGrpcMasterServer(const std::string &master_address) {
  if (master_async_server_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: Master gRPC server is already running";
  }
  SSLConfig ssl_config;
  ssl_config.use_ssl = false;
  master_async_server_ = std::make_shared<MasterGrpcServer>(dispatcher_);
  return master_async_server_->Start(master_address, ssl_config, gRpcMaxMBMsgSize, "Master");
}

Status Server::StartRestfulServer(const std::string &socket_address, const SSLConfig &ssl_config, int max_msg_mb_size,
                                  int time_out_second) {
  return restful_server_.Start(socket_address, ssl_config, max_msg_mb_size, time_out_second);
}

void Server::Clear() {
  MSI_LOG_INFO << "Server start to clean";
  dispatcher_->Clear();
  restful_server_.Stop();
  if (master_async_server_) {
    master_async_server_->Stop();
    master_async_server_ = nullptr;
  }
  if (grpc_async_server_) {
    grpc_async_server_->Stop();
    grpc_async_server_ = nullptr;
  }
  MSI_LOG_INFO << "Server end to clean";
}

Server::Server() = default;

Server &Server::Instance() {
  static Server server;
  return server;
}

bool Server::OnlyModelStage(const std::string &servable_name) { return dispatcher_->OnlyModelStage(servable_name); }

Server::~Server() = default;
}  // namespace serving
}  // namespace mindspore
