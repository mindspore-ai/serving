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

#include "common/grpc_server.h"

namespace mindspore::serving {

Status GrpcServer::Start(std::shared_ptr<grpc::Service> service, const std::string &ip, uint32_t grpc_port,
                         int max_msg_mb_size, const std::string &server_tag) {
  service_ = service;
  Status status;
  if (in_running_) {
    return INFER_STATUS_LOG_ERROR(SYSTEM_ERROR) << "Serving Error: " << server_tag << " server is already running";
  }

  std::string server_address = ip + ":" + std::to_string(grpc_port);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  // Set the port is not reuseable
  auto option = grpc::MakeChannelArgumentOption(GRPC_ARG_ALLOW_REUSEPORT, 0);
  grpc::ServerBuilder serverBuilder;
  serverBuilder.SetOption(std::move(option));
  if (max_msg_mb_size > 0) {
    serverBuilder.SetMaxReceiveMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
  }
  serverBuilder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  serverBuilder.RegisterService(service.get());
  server_ = serverBuilder.BuildAndStart();

  if (server_ == nullptr) {
    return INFER_STATUS_LOG_ERROR(FAILED) << "Serving Error: " << server_tag
                                          << " server start failed, create server failed, address " << server_address;
  }

  auto grpc_server_run = [this, server_address, server_tag]() {
    MSI_LOG(INFO) << server_tag << " server start success,  listening on " << server_address;
    std::cout << "Serving: " << server_tag << " server start success, listening on " << server_address << std::endl;
    server_->Wait();
  };

  grpc_thread_ = std::thread(grpc_server_run);
  in_running_ = true;
  return SUCCESS;
}

void GrpcServer::Stop() {
  if (in_running_) {
    server_->Shutdown();
    grpc_thread_.join();
  }
  in_running_ = false;
}

}  // namespace mindspore::serving
